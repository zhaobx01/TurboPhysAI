#!/usr/bin/env bash
set -euo pipefail

# H20 日志中的训练参数：8 卡、global batch 32、10000 步；只保存最终模型。
export PHYS_ROOT=/workspace/TurboPhysAI/TurboPhysAI
export GROOT_OPT="$PHYS_ROOT/turbo_physai/optimizations/models/groot_optimization"
export GROOT_MODEL=/workspace/TurboPhysAI/model/Isaac-GR00T
export DATA_ROOT=/workspace/data
export PYTHONPATH="$PHYS_ROOT:$GROOT_OPT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"

export NUM_GPUS=8 GLOBAL_BATCH_SIZE=32 MAX_STEPS=10000
export DATALOADER_NUM_WORKERS=4 SHARD_SIZE=1024 NUM_SHARDS_PER_EPOCH=100000
export EPISODE_SAMPLING_RATE=0.1 USE_WANDB=0
unset GR00T_DIT_SDPA_MODE TURBO_PHYSAI_DISABLE_TORCH_COMPILE
unset DS_WEIGHTS_ALPHA SAVE_ONLY_MODEL RESUME_FROM_CHECKPOINT

mkdir -p "$DATA_ROOT/outputs"
BENCH_DIR=$(mktemp -d "$DATA_ROOT/outputs/physai-groot-vs-h20-XXXXXX")
export BENCH_DIR
echo "结果目录：$BENCH_DIR"

# 在 AllGather 的 train 包装执行前禁用训练中 checkpoint；保留最终 save_model。
cat > "$BENCH_DIR/launch_no_checkpoint.py" <<'PY'
import os
import runpy
from functools import wraps

from gr00t.experiment.trainer import Gr00tTrainer
from transformers.trainer_utils import SaveStrategy

original_train = Gr00tTrainer.train

@wraps(original_train)
def train_without_checkpoint(self, *args, **kwargs):
    self.args.save_strategy = SaveStrategy.NO
    if int(os.environ.get('RANK', '0')) == 0:
        print('[vs_h20] 已关闭训练中 checkpoint 保存，保留最终模型保存。', flush=True)
    return original_train(self, *args, **kwargs)

Gr00tTrainer.train = train_without_checkpoint
runpy.run_path(
    os.path.join(os.environ['GROOT_MODEL'], 'gr00t/experiment/launch_finetune.py'),
    run_name='__main__',
)
PY

# 记录本次实际使用的优化配置；是否启用由 optimization.yaml 决定。
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

cd "$GROOT_MODEL"
# 每 10 秒检查显存，所有目标设备连续三次低于 5% 后启动。
python3 -m groot_optimization.resource_wait \
  --devices "$NUM_GPUS" --log "$BENCH_DIR/resource-wait.jsonl"

export MASTER_PORT
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')

# 与 H20 日志使用相同训练参数；最终模型仍由训练入口保存。
python3 -m turbo_physai.cli run \
  --disable-numa \
  --log-report \
  --runtime-config "$GROOT_OPT/configs/runtime.yaml" \
  --optimization-config "$GROOT_OPT/configs/optimization.yaml" -- \
  torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    "$BENCH_DIR/launch_no_checkpoint.py" \
    --base_model_path "$DATA_ROOT/checkpoints/GR00T-N1.7-3B-local-cosmos" \
    --dataset_path "$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_h264" \
    --embodiment_tag LIBERO_PANDA \
    --num_gpus "$NUM_GPUS" --output_dir "$BENCH_DIR" \
    --experiment_name libero10_finetune \
    --global_batch_size "$GLOBAL_BATCH_SIZE" --max_steps "$MAX_STEPS" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --learning_rate 1e-4 --weight_decay 1e-5 --warmup_ratio 0.05 \
    --shard_size "$SHARD_SIZE" --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH" \
    --episode_sampling_rate "$EPISODE_SAMPLING_RATE" \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  2>&1 | tee "$BENCH_DIR/train.log"

echo "结果目录：$BENCH_DIR"
