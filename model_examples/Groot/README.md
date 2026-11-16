# GR00T 应用说明

## 1. 模型简介

[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) 是 NVIDIA 的视觉-语言-动作（VLA）模型，使用 Cosmos Reason2 / Qwen3-VL 骨干处理图像和语言，通过动作 DiT 生成机器人动作。本文使用 GR00T N1.7 3B 权重，在 LIBERO 10 数据集上进行微调，训练入口为 `gr00t/experiment/launch_finetune.py`。

## 2. 优化接入基线

TurboPhysAI 基于官方提交 `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 接入，保持模型源码不变。本文使用 OpenCV CPU 解码，仅开启三个兼容 Group，其余优化关闭：

| Group | 作用 |
| --- | --- |
| `groot.compat.training_config` | 检查 TF32 支持情况，并通过 `VLM_MODEL_PATH` 使用本地 Cosmos。 |
| `groot.compat.opencv_video` | 将训练的数据索引帧读取切换为 OpenCV。 |
| `groot.dit.hip_math_sdpa` | HIP 环境下为动作 DiT 使用 Math SDPA，规避已观察到的数值异常。 |

## 3. 构建训练环境

### 3.1 创建容器

容器创建方式参见[安装指南](../../docs/zh/get_started/installation.md#创建容器)。在宿主机执行以下命令，将 `HOST_WORKSPACE` 替换为实际工作目录；该目录挂载为容器内的 `/workspace`：

```bash
set -euo pipefail
image=harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10
export HOST_WORKSPACE=/path/to/groot-workspace
mkdir -p "$HOST_WORKSPACE"

docker pull "$image"
docker run -dit \
  --network=host \
  --name=physai_groot \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --shm-size=128G \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  --workdir=/workspace \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_WORKSPACE":/workspace \
  --entrypoint /bin/bash \
  "$image"

docker exec -it physai_groot bash
```

**复用本地数据和权重：** 以下宿主机路径沿用原容器创建命令，使用前需在宿主机确认可访问。在 `docker run` 命令的 `--entrypoint` 之前增加对应挂载参数；若资源已位于 `$HOST_WORKSPACE/data` 下，则无需重复挂载。源码按后文准备，无需单独挂载。

| 资源 | 宿主机位置 |
| --- | --- |
| LIBERO H.264 训练数据 | `/public/home/zhaobx/Groot/data/libero_10_no_noops_1.0.0_lerobot_h264` |
| GR00T 起始权重 | `/public/home/zhaobx/Groot/data/checkpoints/GR00T-N1.7-3B-local-cosmos` |
| Cosmos 骨干权重 | `/public/home/zhaobx/Groot/data/checkpoints/nvidia/Cosmos-Reason2-2B` |

```text
  --mount type=bind,src=/public/home/zhaobx/Groot/data/libero_10_no_noops_1.0.0_lerobot_h264,dst=/workspace/data/libero_10_no_noops_1.0.0_lerobot_h264,readonly \
  --mount type=bind,src=/public/home/zhaobx/Groot/data/checkpoints/GR00T-N1.7-3B-local-cosmos,dst=/workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos \
  --mount type=bind,src=/public/home/zhaobx/Groot/data/checkpoints/nvidia/Cosmos-Reason2-2B,dst=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B,readonly \
```

源目录需已存在，已有容器需重新创建才能增加挂载。GR00T 权重目录保留写权限，供第6节备份并修改配置；数据集与 Cosmos 权重只读使用。数据集只读挂载前需有有效的 `meta/stats.json`，否则先以可写方式生成统计；当前数据已满足。

下文均在容器内使用 Bash 执行。检查八卡设备并安装系统工具：

```bash
set -euo pipefail
hy-smi
python3 - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.device_count())
assert torch.version.hip and torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
PY

apt-get update
apt-get install -y git git-lfs ffmpeg
git lfs install
```

### 3.2 准备 TurboPhysAI 与训练依赖

将包含 GR00T 优化子包的 TurboPhysAI 源码放到 `/workspace/TurboPhysA`，框架安装见[安装说明](../../docs/zh/get_started/installation.md)。

```bash
export PHYS_ROOT=/workspace/TurboPhysA
export GROOT_OPT="$PHYS_ROOT/turbo_physai/optimizations/models/groot_optimization"
export GROOT_MODEL="$PHYS_ROOT/model/Isaac-GR00T-official"
export DATA_ROOT=/workspace/data
export HF_HOME="$DATA_ROOT/cache/huggingface"
export PYTHONUNBUFFERED=1
export GROOT_CONFIG_DIR="$DATA_ROOT/configs/groot-opencv-baseline"
test -f "$GROOT_OPT/configs/recipe.yaml"
mkdir -p "$GROOT_CONFIG_DIR" "$DATA_ROOT/outputs" "$DATA_ROOT/logs"
```

镜像提供 DTK 加速栈，额外依赖见 [requirements.txt](requirements.txt)。先锁定厂商包版本再安装，避免替换加速组件；不使用官方 `uv sync`，其 Python/CUDA 依赖与本文 DTK 环境不同。

```bash
python3 - <<'PY' > "$GROOT_CONFIG_DIR/vendor-constraints.txt"
from importlib.metadata import version, PackageNotFoundError
for name in ("torch", "torchvision", "torchaudio", "deepspeed", "flash-attn", "triton", "torchcodec"):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        if name not in ("torchaudio", "torchcodec"):
            raise
PY

python3 -m pip install --dry-run \
  -r "$PHYS_ROOT/model_examples/Groot/requirements.txt" \
  -c "$GROOT_CONFIG_DIR/vendor-constraints.txt"
# 确认安装计划不会替换上述厂商加速包后执行。
python3 -m pip install \
  -r "$PHYS_ROOT/model_examples/Groot/requirements.txt" \
  -c "$GROOT_CONFIG_DIR/vendor-constraints.txt"

# 安装独立的优化子包；依赖已由上一步准备。
python3 -m pip install --no-deps -e "$GROOT_OPT"
export PYTHONPATH="$PHYS_ROOT:$GROOT_OPT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"

python3 -m turbo_physai.cli --help
```

官方入口会导入 W&B，因此仍需安装；训练时通过 `USE_WANDB=0` 关闭上报。`pytest` 也是当前厂商版 FlashAttention 的导入依赖。后续命令复用上述路径变量，新终端需重新设置。

## 4. 准备模型源码

在新机器获取官方仓库并锁定基线：

```bash
mkdir -p "$(dirname "$GROOT_MODEL")"
git clone https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT_MODEL"
git -C "$GROOT_MODEL" switch --detach 51d4c89f72fda44cbf77285c6a8114b52676b8a1
```

已有官方源码时，跳过 clone 和切换提交，确认 `git -C "$GROOT_MODEL" rev-parse HEAD` 为上述提交且工作区干净。若出现 `dubious ownership`，执行：

```bash
git config --global --add safe.directory "$GROOT_MODEL"
```

## 5. 准备 LIBERO 10 数据

训练使用 `/workspace/data/libero_10_no_noops_1.0.0_lerobot_h264`。已有完整数据时按第3.1节挂载，跳过本节；否则下载并转码：

```bash
export DATASET_ORIGINAL="$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_original"
export DATASET_DIR="$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_h264"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  --repo-type dataset --revision e1a223d30b896c1613f270a2bfc63d382b3de7e1 \
  --local-dir "$DATASET_ORIGINAL"

# 保留原始AV1副本，转码只在新目录操作。
test ! -e "$DATASET_DIR"
cp -a "$DATASET_ORIGINAL" "$DATASET_DIR"
python3 "$GROOT_MODEL/examples/SimplerEnv/convert_av1_to_h264.py" \
  "$DATASET_DIR/videos" -j 4
cp "$GROOT_MODEL/examples/LIBERO/modality.json" "$DATASET_DIR/meta/modality.json"
```

核对转码结果并更新编码元数据：

```bash
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
source, target = Path(os.environ["DATASET_ORIGINAL"]), Path(os.environ["DATASET_DIR"])
def probe(path):
    args = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,nb_read_frames,width,height,avg_frame_rate",
            "-of", "json", str(path)]
    return json.loads(subprocess.check_output(args))["streams"][0]
videos = list(source.glob("videos/**/*.mp4"))
assert videos, "没有找到源视频"
for original in videos:
    converted = target / original.relative_to(source)
    before, after = probe(original), probe(converted)
    assert after["codec_name"] == "h264", converted
    for key in ("nb_read_frames", "width", "height", "avg_frame_rate"):
        assert before[key] == after[key], (converted, key, before[key], after[key])
info_path = target / "meta/info.json"
info = json.loads(info_path.read_text())
for feature in info["features"].values():
    if feature.get("dtype") == "video":
        feature.setdefault("info", {})["video.codec"] = "h264"
info_path.write_text(json.dumps(info, indent=2) + "\n")
print(f"已核对{len(videos)}个视频并更新编码元数据")
PY
```

目录需包含 `meta/info.json`、`meta/modality.json`、`data/` 和 `videos/`。

## 6. 准备模型权重

权重位置见第3.1节。已有权重时跳过下载，直接设置本地加载配置；否则先在 [Cosmos 模型页](https://huggingface.co/nvidia/Cosmos-Reason2-2B)获得访问资格，再执行：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf auth login
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --local-dir "$DATA_ROOT/checkpoints/GR00T-N1.7-3B-local-cosmos"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  nvidia/Cosmos-Reason2-2B --revision 9ce19a195e423419c349abfc86fd07178b230561 \
  --local-dir "$DATA_ROOT/checkpoints/nvidia/Cosmos-Reason2-2B"
```

资源就绪后，设置本地 Cosmos 路径并更新 GR00T 配置（保留备份）：

```bash
export VLM_MODEL_PATH="$DATA_ROOT/checkpoints/nvidia/Cosmos-Reason2-2B"
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
root = Path(os.environ["DATA_ROOT"]) / "checkpoints/GR00T-N1.7-3B-local-cosmos"
cosmos = Path(os.environ["VLM_MODEL_PATH"])
assert (cosmos / "config.json").is_file()
for filename in ("config.json", "processor_config.json"):
    path = root / filename
    backup = path.with_name(path.name + ".before-local-loading")
    if not backup.exists():
        shutil.copy2(path, backup)
    config = json.loads(path.read_text())
    if filename == "config.json":
        config["model_name"] = str(cosmos)
    else:
        config["processor_kwargs"]["model_name"] = str(cosmos)
    path.write_text(json.dumps(config, indent=2) + "\n")
print("本地Cosmos路径已写入GR00T配置")
PY
```

训练前开启离线模式：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_WANDB=0
```

## 7. 通过 PhysAI 启动训练

### 7.1 生成优化配置

复制仓库 Recipe 生成独立配置，仅开启训练配置兼容、OpenCV 解码和 HIP Math SDPA，其余优化关闭，不修改仓库原配置：

```bash
python3 - <<'PY'
import os
from pathlib import Path
import yaml
source = Path(os.environ["GROOT_OPT"]) / "configs/recipe.yaml"
target = Path(os.environ["GROOT_CONFIG_DIR"])
target.mkdir(parents=True, exist_ok=True)
recipe = yaml.safe_load(source.read_text())
enabled = {"groot.compat.training_config", "groot.compat.opencv_video", "groot.dit.hip_math_sdpa"}
assert enabled <= {g["id"] for g in recipe["optimization_groups"]}
for group in recipe["optimization_groups"]:
    group["enabled"] = group["id"] in enabled
recipe["metadata"]["id"] = "model.groot.opencv-baseline.hcu"
(target / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
PY

cd "$GROOT_MODEL"
python3 -m turbo_physai.cli optimization generate \
  --recipe "$GROOT_CONFIG_DIR/recipe.yaml" \
  --repo "$GROOT_MODEL" \
  --commit 51d4c89f72fda44cbf77285c6a8114b52676b8a1 \
  --output "$GROOT_CONFIG_DIR/optimization.yaml" --force
python3 -m turbo_physai.cli optimization check \
  "$GROOT_CONFIG_DIR/optimization.yaml" --repo "$GROOT_MODEL"
```

启动时通过 `--set VIDEO_DECODER_BACKEND=opencv` 与 OpenCV Group 配套使用。

### 7.2 快速验证（单机八卡）

先确认八卡空闲。运行10步，全局batch32、每卡4，BF16混合精度。复用无权重保存的启动包装，关闭周期checkpoint和最终保存；临时配置及计时文件结束后清理，仅保留训练日志。

```bash
export NUM_GPUS=8 GLOBAL_BATCH_SIZE=32 MAX_STEPS=10
export DATALOADER_NUM_WORKERS=0 USE_WANDB=0
export MASTER_PORT=29517  # 如被占用，换一个空闲端口
unset GR00T_DIT_SDPA_MODE
LOG_FILE="$DATA_ROOT/logs/groot-opencv-smoke-$(date +%Y%m%d-%H%M%S)-$$.log"
mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/outputs"
(
  set -euo pipefail
  SCRATCH_OUTPUT_DIR=$(mktemp -d "$DATA_ROOT/outputs/groot-smoke.XXXXXX")
  trap 'rm -rf -- "$SCRATCH_OUTPUT_DIR"' EXIT
  export PHYS_BENCH_ATTEMPT_DIR="$SCRATCH_OUTPUT_DIR" PHYS_BENCH_TRACE=0
  cd "$GROOT_MODEL"
  python3 -m turbo_physai.cli run --disable-numa --log-report \
    --runtime-config "$GROOT_OPT/configs/runtime.yaml" \
    --optimization-config "$GROOT_CONFIG_DIR/optimization.yaml" \
    --set VIDEO_DECODER_BACKEND=opencv -- \
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
      "$GROOT_OPT/scripts/rank_launcher.py" \
      --base_model_path "$DATA_ROOT/checkpoints/GR00T-N1.7-3B-local-cosmos" \
      --dataset_path "$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_h264" \
      --embodiment_tag LIBERO_PANDA --num_gpus "$NUM_GPUS" \
      --experiment_name libero10_finetune --output_dir "$SCRATCH_OUTPUT_DIR" \
      --global_batch_size "$GLOBAL_BATCH_SIZE" --max_steps "$MAX_STEPS" \
      --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
      --learning_rate 1e-4 --weight_decay 1e-5 --warmup_ratio 0.05 \
      --state_dropout_prob 0.2 \
      --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee "$LOG_FILE"
)
```

日志保存在 `$LOG_FILE`。应显示三项兼容 Group 为 `applied`、其余15项为 `skipped`，训练达到 `10/10`，并提示 `final save_model skipped`。官方入口可能仍输出 `Model saved to`，不代表已写入权重。

### 7.3 单机八卡训练

正式训练使用官方入口，保留checkpoint与最终模型：

```bash
set -o pipefail
export NUM_GPUS=8 GLOBAL_BATCH_SIZE=32 MAX_STEPS=10000
export SAVE_STEPS=1000 DATALOADER_NUM_WORKERS=4 USE_WANDB=0
export MASTER_PORT=29517
unset GR00T_DIT_SDPA_MODE
OUTPUT_DIR="$DATA_ROOT/outputs/groot-opencv-train-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUTPUT_DIR"
cd "$GROOT_MODEL"
python3 -m turbo_physai.cli run --disable-numa --log-report \
  --runtime-config "$GROOT_OPT/configs/runtime.yaml" \
  --optimization-config "$GROOT_CONFIG_DIR/optimization.yaml" \
  --set VIDEO_DECODER_BACKEND=opencv -- \
  bash examples/finetune.sh \
    --base-model-path "$DATA_ROOT/checkpoints/GR00T-N1.7-3B-local-cosmos" \
    --dataset-path "$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_h264" \
    --embodiment-tag LIBERO_PANDA --experiment-name libero10_finetune \
    --output-dir "$OUTPUT_DIR" --state-dropout-prob 0.2 --save-only-model \
  2>&1 | tee "$OUTPUT_DIR/train.log"
```

训练参数为学习率 `1e-4`、weight decay `1e-5`、warmup `0.05`、梯度累积1；每1000步保存，最多保留5份checkpoint，结束时另存最终模型。`--save-only-model` 不保存优化器、调度器和RNG状态，需要精确续训时去掉该参数。

性能优化开关、测速及后台入口见 [GR00T 优化说明](../../turbo_physai/optimizations/models/groot_optimization/README.md)。现有测速入口使用自己的配置和torchcodec，不继承本文的OpenCV配置。
