#!/usr/bin/env bash
set -euo pipefail

# 按当前优化配置测速：8 卡、global batch 768，每轮 40 步，共两轮。
export PHYS_ROOT=/workspace/TurboPhysAI/TurboPhysAI
export GROOT_OPT="$PHYS_ROOT/turbo_physai/optimizations/models/groot_optimization"
export GROOT_MODEL=/workspace/TurboPhysAI/model/Isaac-GR00T
export DATA_ROOT=/workspace/data
export PYTHONPATH="$PHYS_ROOT:$GROOT_OPT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"

export NUM_GPUS=8 GLOBAL_BATCH_SIZE=768 MAX_STEPS=40 DATALOADER_NUM_WORKERS=4
export SAVE_STEPS=1000
unset GR00T_DIT_SDPA_MODE TURBO_PHYSAI_DISABLE_TORCH_COMPILE

mkdir -p "$DATA_ROOT/outputs"
export BENCH_DIR
BENCH_DIR=$(mktemp -d "$DATA_ROOT/outputs/physai-groot-combo-b768-XXXXXX")
echo "结果目录：$BENCH_DIR"

python3 - <<'PY'
import os
from pathlib import Path
import yaml

config = Path(os.environ['GROOT_OPT']) / 'configs' / 'optimization.yaml'
groups = yaml.safe_load(config.read_text())['optimization_groups']
enabled = [group['id'] for group in groups if group.get('enabled', False)]
disabled = [group['id'] for group in groups if not group.get('enabled', False)]
print(f'优化配置：{config}\n已启用 {len(enabled)}/{len(groups)} 个 Group')
print('启用：' + (', '.join(enabled) if enabled else '无'))
print('关闭：' + (', '.join(disabled) if disabled else '无'))
PY

# 每个训练 rank 在 PhysAI 安装 Group 后运行此启动脚本。
cat > "$BENCH_DIR/launch_benchmark.py" <<'PY'
import json
import os
import runpy
import time
from functools import wraps
from pathlib import Path

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import SaveStrategy
from gr00t.experiment.trainer import Gr00tTrainer


class StepMetricsCallback(TrainerCallback):
    """从上一步结束计时，不额外同步设备，与参考报告的计时方式一致。"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.step_seconds = []
        self.last_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.last_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        self.step_seconds.append(now - self.last_time)
        self.last_time = now

    def on_train_end(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'rank': args.process_index,
            'world_size': args.world_size,
            'step_seconds': self.step_seconds,
        }
        if torch.cuda.is_available():
            payload['max_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
            payload['max_memory_reserved_bytes'] = torch.cuda.max_memory_reserved()
        path = self.output_dir / f'rank-{args.process_index:02d}.json'
        path.write_text(json.dumps(payload, indent=2) + '\n')


original_train = Gr00tTrainer.train


@wraps(original_train)
def train_without_checkpoint(self, *args, **kwargs):
    # 仅扩大 save_steps 不足以阻止最后一步的 DeepSpeed checkpoint。
    self.args.save_strategy = SaveStrategy.NO
    self.add_callback(StepMetricsCallback(Path(os.environ['BENCH_ROUND_DIR']) / 'step_metrics'))
    if int(os.environ.get('RANK', '0')) == 0:
        print('[benchmark] save_strategy=no；已启用逐 rank 每步计时。', flush=True)
    return original_train(self, *args, **kwargs)


def skip_final_model_save(self, *args, **kwargs):
    if int(os.environ.get('RANK', '0')) == 0:
        print('[benchmark] 已跳过最终 trainer.save_model()。', flush=True)


Gr00tTrainer.train = train_without_checkpoint
Gr00tTrainer.save_model = skip_final_model_save
runpy.run_path(
    os.path.join(os.environ['GROOT_MODEL'], 'gr00t/experiment/launch_finetune.py'),
    run_name='__main__',
)
PY

cd "$GROOT_MODEL"
for ROUND in 1 2; do
  export BENCH_ROUND_DIR="$BENCH_DIR/round-$ROUND"
  mkdir -p "$BENCH_ROUND_DIR"
  # 每轮开始前要求 card0～card7 连续三次（每 10 秒）显存占用均低于 5%。
  python3 -m groot_optimization.resource_wait \
    --devices "$NUM_GPUS" --log "$BENCH_ROUND_DIR/resource-wait.jsonl"
  export MASTER_PORT
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
  echo "开始第 $ROUND 轮：$BENCH_ROUND_DIR"

  python3 -m turbo_physai.cli run \
    --disable-numa \
    --log-report \
    --runtime-config "$GROOT_OPT/configs/runtime.yaml" \
    --optimization-config "$GROOT_OPT/configs/optimization.yaml" \
    -- torchrun \
      --nproc_per_node="$NUM_GPUS" \
      --master_port="$MASTER_PORT" \
      "$BENCH_DIR/launch_benchmark.py" \
      --base_model_path "$DATA_ROOT/checkpoints/GR00T-N1.7-3B-local-cosmos" \
      --dataset_path "$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_h264" \
      --embodiment_tag LIBERO_PANDA \
      --num_gpus "$NUM_GPUS" \
      --output_dir "$BENCH_ROUND_DIR/out" \
      --global_batch_size "$GLOBAL_BATCH_SIZE" \
      --gradient_accumulation_steps 1 \
      --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
      --max_steps "$MAX_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --save_total_limit 5 \
      --learning_rate 1e-4 \
      --weight_decay 1e-5 \
      --warmup_ratio 0.05 \
      --shard_size 1024 \
      --num_shards_per_epoch 100000 \
      --episode_sampling_rate 0.1 \
      --state_dropout_prob 0.2 \
      --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee "$BENCH_ROUND_DIR/train.log"

  if find "$BENCH_ROUND_DIR/out" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
    echo "错误：第 $ROUND 轮意外生成了 checkpoint，请检查输出目录。" >&2
    exit 1
  fi
done

# 按报告的逐步最慢 rank、稳定性阈值和窗口计算两轮结果。
python3 - <<'PY'
import json
import math
import os
import statistics
from pathlib import Path

root = Path(os.environ['BENCH_DIR'])
batch_size = 768
results = []
for round_number in (1, 2):
    metrics_dir = root / f'round-{round_number}' / 'step_metrics'
    records = [json.loads((metrics_dir / f'rank-{rank:02d}.json').read_text()) for rank in range(8)]
    if {record['rank'] for record in records} != set(range(8)):
        raise RuntimeError(f'第 {round_number} 轮缺少 rank 数据')
    if any(record['world_size'] != 8 for record in records):
        raise RuntimeError(f'第 {round_number} 轮 world_size 不为 8')
    for record in records:
        steps = record['step_seconds']
        if len(steps) != 40 or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in steps
        ):
            raise RuntimeError(f"第 {round_number} 轮 rank {record['rank']} 没有 40 个有效逐步耗时")

    slowest = [max(record['step_seconds'][step] for record in records) for step in range(40)]
    steady = slowest[10:40]
    cv = statistics.pstdev(steady) / statistics.mean(steady) * 100
    drift = (statistics.mean(steady[15:]) / statistics.mean(steady[:15]) - 1) * 100
    step35 = slowest[34]
    mean31_40 = statistics.mean(slowest[30:40])
    result = {
        'round': round_number,
        'valid': cv <= 5 and abs(drift) <= 5,
        'step35_ms': step35 * 1000,
        'step35_samples_per_second': batch_size / step35,
        'mean31_40_ms': mean31_40 * 1000,
        'mean31_40_samples_per_second': batch_size / mean31_40,
        'cv_pct': cv,
        'drift_pct': drift,
    }
    results.append(result)
    print(
        f"第 {round_number} 轮 | 有效={result['valid']} | "
        f"第35步={result['step35_ms']:.3f} ms，{result['step35_samples_per_second']:.3f} samples/s | "
        f"第31～40步={result['mean31_40_ms']:.3f} ms/step，"
        f"{result['mean31_40_samples_per_second']:.3f} samples/s | CV={cv:.3f}% | 漂移={drift:.3f}%"
    )

valid_results = [result for result in results if result['valid']]
if len(valid_results) < 2:
    print('有效轮次不足两轮，不计算跨轮中位数。')
else:
    step35_sps = statistics.median(result['step35_samples_per_second'] for result in valid_results)
    window_sps = statistics.median(result['mean31_40_samples_per_second'] for result in valid_results)
    print(f'第35步吞吐中位数：{step35_sps:.3f} samples/s')
    print(f'第31～40步吞吐中位数：{window_sps:.3f} samples/s')
    print('参考报告对应值：790.621、785.739 samples/s')

(root / 'summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
print(f'结果目录：{root}')
PY
