# GR00T 接入 TurboPhysAI

本目录在官方 Isaac-GR00T 源码上，通过 TurboPhysAI 注入 HCU/DTK 兼容处理和性能优化。完整环境、数据、权重及快速验证流程见 [GR00T 应用说明](../../../../model_examples/Groot/README.md)，已完成的检查和历史训练记录见 [VALIDATION.md](VALIDATION.md)。

## 基线与默认配置

模型源码仅支持官方提交 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`。运行前保持模型工作区干净，不需要修改官方源码。旧 HCU 适配提交 `d9bdcf9` 的结果只作历史参考，不能直接归到官方基线。

仓库 Recipe 和生成配置包含18个 Group，当前默认仅开启：

- `groot.compat.training_config`
- `groot.dit.hip_math_sdpa`

其余16项关闭，包括 OpenCV 和 `pad_156`。默认使用官方 torchcodec 解码和原始输入长度。应用说明中的 OpenCV 示例会单独生成三项兼容配置，不修改仓库默认配置。

## 目录与依赖

| 位置 | 内容 |
| --- | --- |
| `groot_optimization/` | Catalog、兼容处理、模型与优化器替换实现 |
| `configs/recipe.yaml` | 仓库默认 Group 开关 |
| `configs/optimization.yaml` | CLI 生成的模型兼容及替换配置 |
| `configs/.optimization.yaml.generation.json` | 配置生成记录和输入指纹 |
| `configs/runtime.yaml` | 本地 Cosmos、离线模式、MIOpen 和 NUMA 设置 |
| `scripts/` | 训练、测速、GEMM、Profiler 和资源等待工具 |
| `tests/` | CPU 行为测试、真实目标集成测试及设备测试 |

保留镜像提供的 DTK 版 Torch、DeepSpeed、FlashAttention、Triton、Apex 等组件。普通训练依赖清单见应用说明；不要用官方 `uv sync` 替换当前 Python 3.10 / DTK 环境。官方入口直接导入 W&B，即使关闭上报也需要 `wandb==0.23.0`。

以下为当前开发容器的路径，可按部署位置调整：

```bash
export PHYS_ROOT=/workspace/TurboPhysAI/TurboPhysAI
export GROOT_OPT="$PHYS_ROOT/turbo_physai/optimizations/models/groot_optimization"
export GROOT_MODEL=/workspace/TurboPhysAI/model/Isaac-GR00T-official
export PYTHONPATH="$PHYS_ROOT:$GROOT_OPT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pip install -e "${GROOT_OPT}[official]"
```

子包安装使用 pip 默认隔离构建；不要求升级镜像中的系统 setuptools。框架本身的安装见[安装说明](../../../../docs/zh/get_started/installation.md)。

## Group 清单

| Group | 默认 | 作用 |
| --- | --- | --- |
| `groot.compat.training_config` | 开 | 不支持 TF32 时关闭已请求的 TF32；用非空 `VLM_MODEL_PATH` 覆盖本地 Cosmos 路径，并处理训练入口函数别名。 |
| `groot.compat.opencv_video` | 关 | 可选 OpenCV CPU 索引帧读取，处理数据加载器函数别名、帧顺序和重复索引。 |
| `groot.dit.hip_math_sdpa` | 开 | HIP 上为动作 DiT 使用 Math SDPA；`GR00T_DIT_SDPA_MODE=default` 可覆盖此选择。 |
| `groot.qwen.flash_attn_dynamo_disable` | 关 | 将 Flash Attention 集成调用排除在 Dynamo 图之外。 |
| `groot.qwen.disable_kv_cache` | 关 | 未显式指定 `use_cache` 且无已有 KV 时关闭隐式缓存。 |
| `groot.data.pad_156` | 关 | 在输入准备阶段将文本左侧补齐至156；拒绝超长或不支持的输入。 |
| `groot.qwen.fast_pos_embed_interpolate` | 关 | CPU 构造并缓存位置插值元数据，设备侧保留嵌入查表。 |
| `groot.qwen.rot_pos_emb` | 关 | CPU 构造、缓存视觉旋转位置坐标并批量搬运。 |
| `groot.qwen.get_rope_index` | 关 | 将多模态位置索引计算移到 CPU，结果返回原设备。 |
| `groot.compile.qwen_vision_block` | 关 | 编译视觉 Block 的 `forward`，依赖 Flash Attention 边界隔离。 |
| `groot.compile.dit` | 关 | 编译 `DiT.forward` 和 `AlternateVLDiT.forward`。 |
| `groot.compile.vl` | 关 | 编译整个 `SelfAttentionTransformer.forward`。 |
| `groot.data.pixel_values_bf16` | 关 | DeepSpeed BF16 训练下，在 collator 中提前转换 CPU 图像张量。 |
| `groot.optim.apex_fused_adam` | 关 | 通过 Transformers 选择 Apex FusedAdam，要求多张量扩展可用。 |
| `groot.zero2.grad_reduce_overlap` | 关 | 在匹配的 DTK DeepSpeed 上优化梯度桶归约重叠。 |
| `groot.compile.qwen_text_layer` | 关 | 编译文本 Decoder Layer 的 `forward`，依赖 Flash Attention 隔离和 KV cache 控制。 |
| `groot.zero2.owner_reduce` | 关 | 按梯度 owner 选择 ReduceScatter 或定向 Reduce，依赖梯度归约重叠。 |
| `groot.zero2.allgather_pipeline` | 关 | 分组更新优化器并异步提交 AllGather。 |

编译默认模式为 `max-autotune-no-cudagraphs`，可通过 Group 的 `options.mode` 调整；`TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1` 会禁用编译包装。没有新增 `dit_blocks` 或完整模块级 `text_layers`，也没有独立的 channels-last Group。

`pad_156` 会改变输入形状，已有测试表明它可能改变 VL 自注意力输出、loss 和梯度，不能视为严格数值等价。ZeRO-2 优化会检查 Torch/DeepSpeed 版本及运行条件；AllGather 流水默认要求8个数据并行 rank、BF16、梯度累积1、无 CPU offload，并限制优化器类型，不支持周期 checkpoint 或断点恢复。具体限制以运行时检查为准。

## 配置生成与运行环境

修改仓库 Recipe 或 Catalog 后，重新生成配置及记录；不要手改源码哈希。禁用项不采集目标证据，重新启用也需要重新生成。

```bash
cd "$GROOT_MODEL"
python3 -m turbo_physai.cli optimization generate \
  --recipe "$GROOT_OPT/configs/recipe.yaml" \
  --repo "$GROOT_MODEL" \
  --commit 51d4c89f72fda44cbf77285c6a8114b52676b8a1 \
  --output "$GROOT_OPT/configs/optimization.yaml" --force
python3 -m turbo_physai.cli optimization check \
  "$GROOT_OPT/configs/optimization.yaml" --repo "$GROOT_MODEL"
```

Git 若报 `dubious ownership`，先仅信任该模型目录：`git config --global --add safe.directory "$GROOT_MODEL"`。临时实验可另存 Recipe 和生成配置，避免影响共享配置。

`runtime.yaml` 默认使用 `/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B`、关闭在线 W&B 上报、开启 Hugging Face 离线模式，设置 `MIOPEN_FIND_MODE=1` 和 `PYTORCH_MIOPEN_SUGGEST_NHWC=1`，不进行 PhysAI NUMA 绑定。新机器可用 CLI 的 `--set` 覆盖路径。

OpenCV 需要同时启用对应 Group、重新生成配置，并在运行时传入 `--set VIDEO_DECODER_BACKEND=opencv`；只设置其中一项不会完成切换。OpenCV 分支处理索引帧，不使用 torchcodec 专用 `decoder_kwargs`。训练时显式传入 `--optimization-config` 和 `--runtime-config`，完整示例见应用说明。

`applied` 表示注入成功，不代表对应路径已执行、数值等价或性能提高。

## 训练与测速入口

| 脚本 | 行为 |
| --- | --- |
| `scripts/run_train__vsH20_b32_10000.sh` | 8卡、全局batch32、10000步，读取共享优化配置；关闭周期checkpoint，保留最终模型。 |
| `scripts/run_TestSpeed_benchmark_40.sh` | 8卡、全局batch960，两轮40步测速；读取共享优化配置，不保存权重。 |
| `scripts/run_benchmark_b960.sh` | 后台启动 `benchmark.py`，默认22方案，每方案两轮40步测速和一轮独立40步Trace。 |
| `scripts/rank_launcher.py` | 各rank的计时、保存控制和可选Profiler，由运行器调用。 |
| `scripts/resource_wait.py` | 共用资源等待工具，支持直接运行，位于 `scripts/` 而非优化实现包内。 |
| `scripts/run_gemm_workflow_b960.sh` | 独立的GEMM采集、DTK选优和两轮顺序测速；需显式指定方案，不采Trace。 |

前两个 Shell 脚本仍使用当前工作区的默认路径；`GROOT_MODEL` 支持外部覆盖，但 `PHYS_ROOT` 在脚本内固定，迁移部署时需检查。前台训练脚本如需断开终端后继续运行，应另用 `nohup` 等方式启动。

### 22方案测速

`benchmark.py` 从方案定义和Catalog依赖生成独立配置，不修改仓库 Recipe。基线不补齐；`pad_156` 单项和 `all_on` 包含固定补齐，其他方案不会自动加上该项。每个所谓“单项”仍可能包含必要依赖，实际Group清单写入报告。

```bash
# 仅检查方案，不启动训练
python3 "$GROOT_OPT/scripts/benchmark.py" --output /tmp/groot-preview --dry-run
# 只测试基线和固定补齐；省略 --cases 则运行全部22方案
GR00T_BENCH_OUTPUT=/workspace/data/outputs/groot-b960-selected \
  bash "$GROOT_OPT/scripts/run_benchmark_b960.sh" --cases baseline pad_156
# 使用相同方案和结果目录续跑；耗尽重试的任务需 --retry-failed
GR00T_BENCH_OUTPUT=/workspace/data/outputs/groot-b960-selected \
  bash "$GROOT_OPT/scripts/run_benchmark_b960.sh" --cases baseline pad_156 --resume --retry-failed
```

测速入口每35秒检查8卡显存，连续4次每卡严格低于5%后放行；每次测速、Trace和重试都执行检查。查询失败或设备忙时清零。其他训练及GEMM入口默认每30秒检查、连续3次通过。等待日志包含累计时长。

测速统计每步8个rank的最大墙钟耗时，不额外同步设备；记录第35步和第31–40步均值，两个成功轮次分别计算后取中位数。CV和漂移只展示，不剔除成功轮次；单轮成功明确标记。Trace单独运行，采集第35步后停止Profiler，训练继续到40步，不参与测速统计。默认共享编译缓存，不在轮次间清理。

正常失败自动重试一次，两次失败后继续后续任务；用户中断不自动重试。后台入口使用锁和进程组管理；结果目录保存配置、尝试历史、日志、计时、显存采样、Trace及持续更新的 `results.json`、`report.md`。续跑核对源码、模型、环境和配置指纹，移动文件或更新实现后可能拒绝旧目录，不能手改历史指纹强行混入新结果。

### GEMM工作流

```bash
# 只查看计划
bash "$GROOT_OPT/scripts/run_gemm_workflow_b960.sh" plan --cases combo_runtime all_on
# 确认计划后后台执行
bash "$GROOT_OPT/scripts/run_gemm_workflow_b960.sh" start \
  --output /workspace/data/outputs/groot-gemm-b960 --cases combo_runtime all_on
```

工作流自动把基线放在首位，所有采集和测速方案统一加入 `pad_156`。采集固定使用 `combo_obr_dit_vl_text_allgather`，8卡、batch960、40步；测速只做两轮，不采Trace。选优使用 `update_target_config=0`，归档后通过 `ROCBLAS_TENSILE_LIBPATH` 加载，不覆盖已安装rocBLAS库。每阶段失败重试一次，耗尽后停止。

DTK选优仍依赖外部 `/workspace/Isaac-GR00T/getting_started/optimization/dcu_gemm_autotune.sh`，可用 `PHYS_GEMM_AUTOTUNE` 指定其他位置；该工具不会随单独克隆本优化子包提供。`status`、`stop`、`resume` 管理同一结果目录的后台任务。

## 验证

```bash
cd "$PHYS_ROOT"
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
python3 -m pytest "$GROOT_OPT/tests" -m 'not hcu' -q -rs
for script in "$GROOT_OPT"/scripts/*.sh; do bash -n "$script"; done
```

最新结果、跳过原因及历史实验边界见 [VALIDATION.md](VALIDATION.md)。
