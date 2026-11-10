#!/usr/bin/env bash
# 等 GPU 空闲后再启动 MapTRv2 8 卡训练，并在后台运行（断开终端不会挂断）。
#
#   bash run_train_when_idle.sh            # 提交到后台，立即返回
#   DRY_RUN=1 bash run_train_when_idle.sh  # 只演练：打印命令，不等待、不训练
#
# 行为：
#   1. 每隔 INTERVAL 秒（默认 30）检查一次 8 张卡的利用率；
#   2. 所有卡的利用率都低于 THRESHOLD%（默认 5）才启动训练，否则继续等；
#      每次重试前都会重新等待，避免和刚释放的资源抢卡；
#   3. 脚本以 setsid + nohup 方式脱离终端，输出写入日志文件。
#
# 日志布局（与 diag_one2many/run_eval6.sh 对齐）：
#   <MODEL_ROOT>/<work_dir 的父目录>/launcher_logs/
#     train_when_idle_<时间戳>.log          启动器输出（等待/重试/汇总）
#     train_when_idle_latest.log -> 上面那个
#     train_when_idle_<时间戳>/
#       train.log          本次训练的完整 stdout/stderr
#       train_try<N>.log   第 N 次尝试失败时的现场（成功后不生成）
#       exit_code          最终退出码
#       attempts           实际尝试次数
#       summary.txt        各 epoch 验证的 NuscMap_chamfer/mAP
#   work_dir 下的 mmcv 日志（<时间戳>.log、.log.json、tf_logs/）保持不变。
#
# 查看 / 停止：
#   tail -f <启动器日志>                    # 启动时会打印路径
#   kill -TERM -- -$(cat /tmp/maptrv2_train_when_idle.pid)   # 结束整个进程组（含训练）
#
# 注意：下面的默认值来自本机验证环境（/workspace 下的路径、ens19f0 网卡、
# 29501 端口等），属于示例值，不是通用约定。在其他机器上运行时请通过环境
# 变量覆盖 MODEL_ROOT/TURBO_SRC/CFG_DIR/MAP_ANN/WORK_DIR/SOCKET_IFNAME/
# MASTER_PORT/TRAIN_ENTRY，或在启动前直接修改本文件。
set -uo pipefail

# ---------- 可通过环境变量覆盖的参数 ----------
THRESHOLD=${THRESHOLD:-5}          # GPU 利用率阈值（%），所有卡都要低于它
INTERVAL=${INTERVAL:-30}        # 两次检查的间隔（秒）
SAMPLES=${SAMPLES:-3}              # 每次检查的采样次数，全部低于阈值才算空闲
SAMPLE_GAP=${SAMPLE_GAP:-5}        # 采样间隔（秒）
NPROC=${NPROC:-8}
DRY_RUN=${DRY_RUN:-0}
MAX_RETRY=${MAX_RETRY:-2}          # 每个训练最多重试次数（仅对可重试的瞬时故障）
TIMEOUT=${TIMEOUT:-14400}          # 单次训练超时（秒），默认 4 小时；超时不重试
RETRY_PATTERN=${RETRY_PATTERN:-'out of memory|OutOfMemory|Bus error|NCCL error|ncclInternalError|ncclSystemError|Address already in use|Connection reset|DistBackendError|VMFault|Invalid address access'}

MODEL_ROOT=${MODEL_ROOT:-/workspace/model/MapTrv2}
TURBO_SRC=${TURBO_SRC:-/workspace/TurboPhysAI}
CFG_DIR=${CFG_DIR:-/workspace/TurboPhysAI/turbo_physai/optimizations/models/maptrv2/configs}
MAP_ANN=${MAP_ANN:-/workspace/MapTRv2/anno_pkls/nuscenes_map_anns_val.json}
WORK_DIR=${WORK_DIR:-work_dirs/maptrv2_optimized_8card}
TRAIN_ENTRY=${TRAIN_ENTRY:-tools/train.py}   # 训练入口；短跑冒烟可指向 probe_train.py（配合 PROBE_MAX_ITERS）
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-12}
SOCKET_IFNAME=${SOCKET_IFNAME:-ens19f0}
MASTER_PORT=${MASTER_PORT:-29501}

PID_FILE=${PID_FILE:-/tmp/maptrv2_train_when_idle.pid}
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 第一阶段：脱离终端 ----------
if [[ -z "${_TRAIN_DAEMON:-}" ]]; then
  if [[ -f $PID_FILE ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "已有一个实例在运行 (PID $(cat "$PID_FILE"))，先停止它：" >&2
    echo "  kill -TERM -- -$(cat "$PID_FILE")" >&2
    exit 1
  fi
  [[ -d $MODEL_ROOT ]] || { echo "模型目录不存在: $MODEL_ROOT" >&2; exit 1; }

  if [[ -z ${LOG_DIR:-} ]]; then
    WORK_PARENT=$(dirname "$WORK_DIR")
    [[ $WORK_PARENT == /* ]] && LOG_DIR="$WORK_PARENT/launcher_logs" \
                              || LOG_DIR="$MODEL_ROOT/$WORK_PARENT/launcher_logs"
  fi
  mkdir -p "$LOG_DIR"

  RUN_TS=$(date +%Y%m%d_%H%M%S)
  export LAUNCH_LOG="$LOG_DIR/train_when_idle_${RUN_TS}.log"
  export RUN_DIR="$LOG_DIR/train_when_idle_${RUN_TS}"
  mkdir -p "$RUN_DIR"
  ln -sfn "$LAUNCH_LOG" "$LOG_DIR/train_when_idle_latest.log"

  _TRAIN_DAEMON=1 setsid nohup bash "$SCRIPT_PATH" "$@" \
    >> "$LAUNCH_LOG" 2>&1 < /dev/null &
  sleep 1
  echo "已提交到后台。关闭终端不会中断。"
  echo "  启动器日志: $LAUNCH_LOG"
  echo "  训练日志:   $RUN_DIR/train.log"
  echo "  PID:        $(cat "$PID_FILE" 2>/dev/null || echo '(启动中)')"
  echo "  查看:       tail -f $LAUNCH_LOG"
  echo "  停止:       kill -TERM -- -\$(cat $PID_FILE)"
  exit 0
fi

# ---------- 第二阶段：后台守护进程 ----------
echo $$ > "$PID_FILE"
# 直接以 _TRAIN_DAEMON=1 运行（没经过第一阶段）时兜底
if [[ -z ${RUN_DIR:-} ]]; then
  RUN_DIR="$MODEL_ROOT/$(dirname "$WORK_DIR")/launcher_logs/train_when_idle_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RUN_DIR"
fi
cleanup() { rm -f "$PID_FILE"; }
trap cleanup EXIT
on_term() {
  trap - TERM INT
  log "收到终止信号，结束整个进程组"
  kill -TERM -- -$$ 2>/dev/null
  exit 143
}
trap on_term TERM INT

# 返回所有卡中最高的利用率；读取失败或卡数不足时返回非 0
max_gpu_util() {
  local values count
  values=$(rocm-smi --showuse 2>/dev/null | awk '/HCU use \(%\)/ {print $NF}')
  count=$(grep -c . <<< "$values")
  [[ $count -ge $NPROC ]] || return 1
  sort -g <<< "$values" | tail -1
}

# 连续 SAMPLES 次采样，最高利用率都低于阈值才算空闲；busy 时把最高值写到 LAST_MAX
gpus_idle() {
  local i util
  LAST_MAX=""
  for ((i = 1; i <= SAMPLES; i++)); do
    util=$(max_gpu_util) || { LAST_MAX="读取失败"; return 1; }
    LAST_MAX=$util
    awk -v u="$util" -v t="$THRESHOLD" 'BEGIN { exit !(u < t) }' || return 1
    ((i < SAMPLES)) && sleep "$SAMPLE_GAP"
  done
  return 0
}

# 等到 GPU 空闲；DRY_RUN 下不等待，方便直接看将执行的命令
wait_idle() {
  [[ $DRY_RUN == 1 ]] && return 0
  while true; do
    if gpus_idle; then
      log "GPU 空闲 (最高利用率 ${LAST_MAX}%)，开始训练"
      return 0
    fi
    log "GPU 繁忙或不可读 (最高利用率 ${LAST_MAX}%)，${INTERVAL}s 后重试"
    sleep "$INTERVAL" & wait $!
  done
}

log "启动等待: 所有 ${NPROC} 张卡利用率 < ${THRESHOLD}% 才开始训练，每 ${INTERVAL}s 检查一次"
log "启动器日志: ${LAUNCH_LOG:-终端}"
log "训练日志:   $RUN_DIR/train.log"

# ---------- 训练命令 ----------
[[ -d $MODEL_ROOT ]] || { log "模型目录不存在: $MODEL_ROOT"; exit 1; }
export PYTHONPATH="$TURBO_SRC:${PYTHONPATH:-}"

CMD=(
  turbo-physai run
    --optimization-config "$CFG_DIR/optimization.yaml"
    --runtime-config "$CFG_DIR/runtime.yaml"
    --log-report
    --set "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME"
    torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" "$TRAIN_ENTRY"
      projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py
      --launcher pytorch
      --work-dir "$WORK_DIR"
      --cfg-options
        "data.val.map_ann_file=$MAP_ANN"
        "data.test.map_ann_file=$MAP_ANN"
        "data.samples_per_gpu=$SAMPLES_PER_GPU"
)

if [[ $DRY_RUN == 1 ]]; then
  log "[DRY_RUN] 将执行: $(printf '%q ' "${CMD[@]}")"
  exit 0
fi

attempt=1
while :; do
  wait_idle   # 每次（含重试）启动前都重新等待，避免和刚释放的资源抢卡
  log "第 ${attempt}/$((MAX_RETRY + 1)) 次尝试"
  log "执行: $(printf '%q ' "${CMD[@]}")"
  ( cd "$MODEL_ROOT" && timeout "$TIMEOUT" "${CMD[@]}" ) > "$RUN_DIR/train.log" 2>&1 &
  wait $!
  rc=$?
  log "训练结束，退出码 = $rc（train.log: $RUN_DIR/train.log）"
  (( rc == 0 )) && break

  # 保留每次失败的现场
  cp "$RUN_DIR/train.log" "$RUN_DIR/train_try${attempt}.log"
  if (( rc != 124 )) && (( attempt <= MAX_RETRY )) && grep -q -E "$RETRY_PATTERN" "$RUN_DIR/train.log"; then
    log "命中可重试故障: $(grep -o -E "$RETRY_PATTERN" "$RUN_DIR/train.log" | head -1)，重新排队"
    attempt=$((attempt + 1))
    sleep 20
    continue
  fi
  (( rc == 124 )) && log "超时(${TIMEOUT}s)，不重试"
  (( rc != 124 )) && ! grep -q -E "$RETRY_PATTERN" "$RUN_DIR/train.log" && log "非瞬时故障，不重试"
  break
done

echo $rc > "$RUN_DIR/exit_code"
echo $attempt > "$RUN_DIR/attempts"

log "================ 验证结果 ================"
grep -hE "Epoch\(val\) \[[0-9]+\].*NuscMap_chamfer/mAP:" "$RUN_DIR/train.log" \
  | sed -E 's/.*(Epoch\(val\) \[[0-9]+\]).*(NuscMap_chamfer\/mAP: [0-9.]+).*/  \1  \2/' \
  | tee "$RUN_DIR/summary.txt"
[[ -s "$RUN_DIR/summary.txt" ]] || log "（未解析到验证 mAP，见 $RUN_DIR/train.log）"
log "结果目录: $RUN_DIR"

exit $rc
