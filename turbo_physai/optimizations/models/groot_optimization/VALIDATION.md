# GR00T 验证记录

更新日期：2026-11-16。本文区分当前代码检查和已有训练记录，不将旧测试计数或历史性能视为当前代码重新实测的结果。

## 当前代码与配置

- 模型基线：官方 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`。
- 当前官方工作树：`/workspace/TurboPhysAI/model/Isaac-GR00T-official`，检查时提交匹配、工作区干净。
- 仓库配置共18项，默认只启用 `groot.compat.training_config`、`groot.dit.hip_math_sdpa`；其余16项均关闭，包括OpenCV和固定补齐。
- 应用说明中的OpenCV示例另行生成三项兼容配置，不改变仓库默认开关。
- `resource_wait.py` 已移到 `scripts/`，训练、测速、测试引用和源码指纹已同步调整。

当前路径来自开发容器；[应用说明](../../../../model_examples/Groot/README.md)中的 `/workspace/TurboPhysA` 是新部署示例，当前源码没有搬迁。

## 本轮自动化验证

本轮未安装或升级依赖，未启动模型训练、GEMM选优或设备测速。

| 检查 | 结果与范围 |
| --- | --- |
| GR00T CPU回归 | **164 passed、3 skipped、5 deselected**，28条依赖弃用或Profiler等警告，没有失败。 |
| DataLoader tensor IPC | 1项跳过：沙箱不允许所需Unix socket。 |
| 双rank Gloo | 2项跳过：沙箱不允许Gloo socket初始化。 |
| `hcu` 设备测试 | 5项由 `-m 'not hcu'` 排除，未执行。 |
| 保存控制与异常传播 | 已包含在上述回归中，验证周期保存关闭、最终保存跳过，以及原训练异常不被吞掉。 |
| 默认配置及生成记录 | `optimization check` 通过；18项、默认2项，官方SHA匹配。 |
| 脚本入口 | 全部Shell脚本 `bash -n` 通过；新位置的资源等待CLI帮助可运行。 |
| 调度计划 | 22方案dry-run及GEMM的指定方案plan通过，未启动训练或选优。 |

命令如下；需要复现3项IPC测试时，应在允许本机socket的环境执行，不能把跳过计作通过：

```bash
export PHYS_ROOT=/workspace/TurboPhysAI/TurboPhysAI
export GROOT_OPT="$PHYS_ROOT/turbo_physai/optimizations/models/groot_optimization"
export GROOT_MODEL=/workspace/TurboPhysAI/model/Isaac-GR00T-official
export PYTHONPATH="$PHYS_ROOT:$GROOT_OPT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"
cd "$PHYS_ROOT"
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
PYTHONDONTWRITEBYTECODE=1 \
python3 -m pytest "$GROOT_OPT/tests" -m 'not hcu' -q -rs
```

本轮临时测试日志为 `/tmp/groot-doc-rewrite-cpu-tests.log`；它不是随仓库交付的永久产物。覆盖范围包含兼容包装、真实目标应用/回滚、18项组合准备、输入补齐、优化器及通信行为、22方案定义、重试续跑、报告、资源等待和CPU合成Profiler窗口。本轮未重新运行全仓其他模型及算子测试，不声明全仓回归通过。

## 已有官方基线实验

以下是现有训练和测速记录的复核，不是2026-11-16重新训练。模型权重、数据、Trace和运行日志均在仓库外，单独克隆代码不会获得这些产物。

### 初次兼容基线测速

结果目录：`/workspace/data/outputs/physai-groot-b960-official-baseline-20261111`。

8卡、全局batch960，两轮40步测速和一轮独立Trace；仅开启两项兼容Group，不启用 `pad_156`。历史报告记录第35步吞吐两轮中位数为489.475 samples/s，第31–40步为490.524 samples/s。该记录属于当时代码、共享缓存及设备环境，不是当前所有方案的统一基线。

### 22方案测速与Trace

当前归档目录：`/workspace/data/benchmarks/all-opyimization-new/physai-groot-b960-20261111-210920-2266429`。目录已由 `outputs/` 移动，部分历史JSON仍保留原路径用于追溯，不应改写旧指纹或据此直接续跑。

本轮复核 `manifest.json`、`results.json` 和报告：

- manifest记录官方完整SHA；22个方案，全部66个逻辑运行成功。
- 包含44个成功测速运行、22个成功Trace运行，共69次尝试；3次失败保留记录，后续重试成功。
- 固定8卡、全局batch960、每轮40步，Trace不参与测速统计。
- 严格兼容基线不补齐；`pad_156` 单项和 `all_on` 启用固定补齐，各方案完整依赖见归档Recipe。

同批报告中的部分吞吐如下，均为两个成功轮次分别计算后的中位数：

| 方案 | 第35步 samples/s | 第31–40步 samples/s |
| --- | ---: | ---: |
| baseline | 435.631 | 455.334 |
| combo_obr_dit_vl_text_allgather | 636.933 | 38.530 |
| all_on | 843.830 | 839.786 |

基线两轮第35步为377.481和493.782 samples/s，存在明显轮间差异；若干编译方案在31–40步窗口出现长耗时。不能将第35步的提升直接解释为稳态、全程或纯硬件收益，也不能混用初次基线和此批结果计算提升。此次复核未重新解析全部大体积Trace，详细时间线结论应回到原始Trace核验。

### 官方兼容基线10000步训练

结果目录：`/workspace/data/outputs/physai-groot-vs-h20-qVA3Ik`。`train.log` 包含完成记录，保存配置和最终模型仍在 `libero10_finetune/`。

| 项目 | 已有日志/产物 |
| --- | --- |
| 规模 | 8卡、全局batch32、10000步、BF16混合精度 |
| Group | 两项兼容Group applied，其余16项关闭；使用torchcodec |
| 起始权重 | `/workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos` |
| Cosmos | `/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B` |
| 数据 | `/workspace/data/libero_10_no_noops_1.0.0_lerobot_h264` |
| 训练耗时 | 3175.9692秒 |
| Trainer全程吞吐 | 100.757 samples/s，3.149 step/s |
| 全程平均loss | 0.15516771545410157 |
| 保存 | 无周期checkpoint；结束后保存两个safetensors分片，合计6910499416字节 |

这是官方兼容基线的长训练完成证据，不是优化全开结果。H20参考日志有每1000步保存checkpoint的开销，双方保存策略不同，不能直接把全程速度差归因于硬件或PhysAI优化。

## 结论与待验证边界

当前已具备官方基线兼容代码、CPU测试、配置生成检查以及历史多卡训练/测速记录，不再使用“官方基线完全未训练”或“默认17项开启”等旧结论。

以下仍不能由现有记录直接证明：

- 所有优化严格数值等价。`pad_156` 已有测试显示会改变VL自注意力输出、loss和梯度。
- 40步loss相对偏差低于5%代表最终精度无影响；它仅是短程训练观察，不代替LIBERO下游评测。
- 新版代码或OpenCV三项配置已重新完成10000步训练；现有长训练使用历史两项兼容配置和torchcodec。
- 最新GEMM编排脚本的真实选优收益；本轮只检查代码/测试，不启动选优。
- 保存后的完整重载推理、仿真成功率、优化全开长期稳定性，以及统一保存策略下的H20公平性能对比。

旧 `d9bdcf9` 基线的结果保留原始归属；当前实现、目录或环境变化后，应建立新结果目录验证，不修改历史产物的指纹。
