# GR00T 接入 PhysAI

## 任务目标与基线

以 Isaac-GR00T 为模型代码基线，将 HCU/DTK 正确性兼容处理和经过验证的性能优化接入 PhysAI，拆分为可独立启用、检查、回滚和验证的 Optimization Group，并交付配置、测试及验证记录。

模型基线固定为 `/workspace/TurboPhysAI/model/Isaac-GR00T` 的 `d9bdcf9686649761440496445dd60c1bbb225ea1`（HCU 适配）。该提交位于待迁移的 Qwen 算法修改、局部编译及后续训练优化之前。模型代码基线与 PhysAI 代码版本、配置和运行环境应分别记录；当前 Group 的数值验证范围见下文。

## 已接入：DiT HIP math SDPA

`groot.dit.hip_math_sdpa` 包装 GR00T 的 DiT `_sdpa_context`：在 HIP 上为该注意力上下文选择 PyTorch math SDPA，退出上下文后恢复原后端设置。在非 HIP 环境或设置 `GR00T_DIT_SDPA_MODE=default` 时，保留模型原有行为。该开关会绕过此兼容处理。

该兼容项参考 GR00T 后续的 HIP 修复。math 后端可能改变性能；它是正确性兼容处理，其性能影响需要实测。

### 当前验证结果

- 未启用该 Group 的一次 1 卡、batch 1、10 步运行中，记录的 `grad_norm` 为 NaN，保存的 `action_head.action_decoder.layer1.W` 含非有限值。该 checkpoint 不可作为有效训练起点。
- 开启 Group 后，1 卡、batch 1、10 步运行的 OptimizationReport 显示 `groot.dit.hip_math_sdpa` 为 `applied`；第 10 步记录的 loss 为 `1.288`，`grad_norm` 为 `1.2960360050201416`，均为有限值。
- 上述运行的 `checkpoint-10` 共检查 1030 个张量，均为有限值。日志和训练记录位于 `/workspace/data/outputs/physai-groot-math-sdpa-20261105-171612/`。
- Group 的后端选择、原行为回退、真实目标替换和恢复及相关配置测试共 14 项通过。

这次短训练仅记录了第 10 步的梯度范数，不能证明每一步梯度有限，也不能替代多卡或长时间训练的数值验证。此前 NaN 的确切根因尚未由这次对照单独证明。

## 后续任务

1. 增加逐步数值检查，并验证多卡和更长训练的 loss、梯度、权重及稳定性。
2. 数值验证通过后，以 `d9bdcf9` 加 `groot.dit.hip_math_sdpa` 作为性能对照配置，记录代码版本、环境、数据、权重、训练参数、吞吐和显存。
3. 对已接入的 Qwen 位置算法、局部编译、数据转换、优化器和 ZeRO-2 Group 逐项验证数值、梯度与性能。
4. 验证 Group 组合，固定 PhysAI 版本与配置，保留 OptimizationReport、测试结果及性能记录。

## 实验性性能 Group

以下 Group 已声明，但尚未完成目标设备的数值、稳定性和性能验证。`recipe.yaml` 仅启用 math SDPA 兼容项，其他 Group 默认关闭，这是唯一随工程交付的配置；启用其他 Group 时修改 Recipe 的 `enabled` 后重新生成。集成测试在临时副本中启用全部 Group，检查目标冲突，不交付第二份配置。修改 Recipe 后必须重新生成配置，以采集对应的源码证据；不要直接修改生成的 YAML。禁用 Group 不生成源码证据，启用前必须重新生成。

位置算法来自 Isaac-GR00T `2838dd8` 及后续 `bba613c` 代码；编译、数据转换、Apex 与 ZeRO-2 实现参考 `bb1cb72` 至 `bba613c` 的工作树。视觉块 forward 级编译与原类级编译并非同一实现。部分 Group 已完成下文所述的小模型设备测试；完整模型的训练行为尚未验证，不能凭安装成功推断运行时生效或数值正确。

| Group | 作用及条件 |
| --- | --- |
| `groot.qwen.fast_pos_embed_interpolate` | 在 CPU 构造位置插值索引，保留设备侧的可训练位置嵌入查表。 |
| `groot.qwen.rot_pos_emb` | 在 CPU 构造视觉旋转位置坐标。 |
| `groot.qwen.get_rope_index` | 多模态输入的位置索引在 CPU 计算后返回原设备。 |
| `groot.qwen.flash_attn_dynamo_disable` | 将 Transformers 的 Flash Attention 集成排除在 Dynamo 图之外。 |
| `groot.compile.qwen_vision_block` | 编译视觉块 `forward`，依赖上一个 Group；这与原仓库的类级编译不等价，须验证 hook、梯度检查点及图行为。 |
| `groot.compile.dit`、`groot.compile.vl` | 分别编译 DiT 与视觉语言注意力模块的 `forward`。 |
| `groot.data.pixel_values_bf16` | 在 DeepSpeed BF16 场景下将 collator 的 CPU 图像数据提前转换为 BF16。 |
| `groot.data.pad_156` | 在 Trainer 输入准备阶段使用 tokenizer 的 pad token 将文本输入左侧补齐到 156；拒绝超长、非左填充掩码和不支持的字段。 |
| `groot.optim.apex_fused_adam` | 在支持条件下通过 Transformers 优化器工厂选择 Apex FusedAdam，需要 Apex 及其多张量扩展。 |
| `groot.zero2.grad_reduce_overlap` | 实验性 ZeRO-2 梯度归约重叠，仅适用于经过校验的 DTK DeepSpeed/Torch 版本；必须先通过真实多卡正确性验证。 |
| `groot.qwen.disable_kv_cache` | 未显式请求缓存且未传入已有缓存时，关闭文本模型隐式 KV cache。 |
| `groot.compile.qwen_text_layer` | 编译 Qwen 文本解码层 `forward`，依赖 Flash Attention 排除和隐式 KV cache 控制。 |
| `groot.zero2.owner_reduce` | GRO 中按分片 owner 归约梯度；不满足调用条件时回退原方法。依赖 GRO。 |
| `groot.zero2.allgather_pipeline` | 重分组优化器，在逐组参数更新后异步提交 AllGather；默认要求 8 个数据并行 rank。 |

编译 Group 可通过 `options.mode` 指定模式，默认为 `max-autotune-no-cudagraphs`；设置 `TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1` 可禁用包装。ZeRO-2 的 `options.strict` 默认为 `true`：不满足运行条件时报错；设为 `false` 才允许回退原实现。配置选项不会替代设备上的正确性和性能验证。

`groot.data.pad_156` 对应原仓库 `GR00T_COMPILE_SEQUENCE_POLICY=pad-156` 的固定左填充策略，由 PhysAI Group 开关控制，不读取该环境变量。它会改变输入形状，当前尚未验证模型输出、梯度或训练数值等价性，也未证明性能收益；超过 156 的输入不会截断，而是报错。

ZeRO-2 一致性检查在 optimizer 构造完成后运行，比较各 rank 的适用性和分片布局；其异步事件生命周期及真实多卡调用顺序尚未验证。使用前必须完成 2 卡和 8 卡逐步梯度、参数更新与异常路径测试。除 ZeRO-2 外，当前 Group 尚未提供完整的运行时激活计数或逐实例日志，OptimizationReport 的 `applied` 只表示替换已安装，不代表运行路径一定触发。

新接入的 Text 编译、OBR 和 AllGather 流水默认关闭。Text 编译已在 HCU 上通过 16 层小模型的输出、梯度和单图检查。OBR 在 2 卡及 8 卡 ZeRO-2 小模型上通过逐步对照并覆盖 reduce-scatter；AllGather 在 8 卡小模型上通过正常更新、溢出和一次同步注入的异常检查。这些测试不能替代完整模型训练及长时间稳定性验证；三者均未完成性能验证。AllGather 使用 `options.group_elements`（默认 100000000）、`options.world_size`（默认 8）和 `options.strict`（默认 true）；要求 BF16、梯度累积为 1，不支持从 checkpoint 恢复或训练中途保存，并仅支持非 fused 的 `torch.optim.AdamW` 或兼容的 Apex FusedAdam。OBR 统计按路径保存在 optimizer 的 `_turbo_physai_obr` 中。`pad_156` 已知会改变 VL 自注意力数值，不应与其他 Group 一同默认启用。

CPU 小模型测试中，16 层文本模型不使用 KV cache 时共用一张 Dynamo 图；使用 cache 时达到 8 次重编译上限。AllGather 若异步请求等待失败，会保留仍在途的请求和张量引用；执行 step 发生异常后拒绝复用该优化器。单个 rank 在其他 rank 提交下一次集合通信之前异常退出仍可能导致后者阻塞，应终止整个分布式作业；异常路径尚需多卡验证。`strict: false` 的回退使用已经重分组的优化器，参数组结构可能与原始优化器不同。

## 使用与验证

`configs/runtime.yaml` 设置当前工作区的本地 Cosmos 路径、离线模式和关闭 W&B，清理视频后端及编译缓存等环境变量，并关闭 PhysAI 自动 NUMA 绑定，以匹配当前参考启动方式。它默认使用 torchcodec；部署环境须有可用的 torchcodec。该文件不设置设备数量、训练步数或 batch size。路径在其他环境中可通过 `--set VLM_MODEL_PATH=/path/to/checkpoint` 覆盖；环境变量由 RuntimeConfig 设置时，使用 `--set` 可以覆盖配置，包括选择 `VIDEO_DECODER_BACKEND=opencv`。若需要自定义缓存路径，也可通过 `--set TORCHINDUCTOR_CACHE_DIR=...` 设置。优化 Group 仍由 OptimizationConfig 控制。

以下示例使用仅启用 math SDPA 的对照配置。ZeRO-2 相关 Group（`grad_reduce_overlap`、`owner_reduce`、`allgather_pipeline`）默认 `strict: true`，单卡、非 DeepSpeed ZeRO-2 环境下若被启用会直接报错而不是静默跳过。示例要求已有 DTK 环境、模型和数据；输出目录应使用新的路径，使用原始权重而不是此前含 NaN 的 checkpoint。

```bash
export PHYS_ROOT=/workspace/TurboPhysAI/TurboPhysAI
export GROOT_OPT="$PHYS_ROOT/turbo_physai/optimizations/models/groot_optimization"
export GROOT_MODEL=/workspace/TurboPhysAI/model/Isaac-GR00T
export PYTHONPATH="$PHYS_ROOT:$GROOT_OPT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"
export NUM_GPUS=1 GLOBAL_BATCH_SIZE=1 MAX_STEPS=10 SAVE_STEPS=1000 DATALOADER_NUM_WORKERS=0
unset GR00T_DIT_SDPA_MODE
cd "$GROOT_MODEL"
python3 -m turbo_physai.cli run --disable-numa --log-report \
  --runtime-config "$GROOT_OPT/configs/runtime.yaml" \
  --set VIDEO_DECODER_BACKEND=opencv \
  --optimization-config "$GROOT_OPT/configs/optimization.yaml" -- \
  bash examples/finetune.sh \
    --base-model-path /workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos \
    --dataset-path /workspace/data/libero_10_no_noops_1.0.0_lerobot_h264 \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /workspace/data/outputs/physai-groot-math-sdpa-smoke \
    --state-dropout-prob 0.2 --save-only-model
```

确认每个训练进程的 OptimizationReport 中该 Group 为 `applied`，检查整个训练过程的 loss、梯度和保存权重均为有限值，再进行多卡及更长训练验证。日志中的最终 loss 或训练进程成功退出不足以证明每一步数值正常。不能用 `python -c` 验证启动注入，因为框架有意跳过此类进程。

可在设备空闲、依赖齐备时复现小模型检查，以下只执行测试，不启动 GR00T 训练：

```bash
cd "$PHYS_ROOT"
python3 -m pytest "$GROOT_OPT/tests/test_compile_text.py" -m hcu -q
python3 -m torch.distributed.run --standalone --nproc_per_node=2 -m pytest \
  "$GROOT_OPT/tests/test_hcu_zero2.py" -k owner -q
python3 -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest \
  "$GROOT_OPT/tests/test_hcu_zero2.py" -q
```

修改 Recipe、Catalog 或模型基线后，在干净的模型工作树重新生成配置：

```bash
cd "$GROOT_MODEL"
python3 -m turbo_physai.cli optimization generate \
  --recipe "$GROOT_OPT/configs/recipe.yaml" --repo "$GROOT_MODEL" \
  --commit "$(git rev-parse HEAD)" --output "$GROOT_OPT/configs/optimization.yaml" --force
```
