# v5 Group A: A1 + A2

`maptrv2.sync_elimination` 以一个原子组安装四个方法替换。任一目标检查失败时不安装整组；安装失败时回退已执行的替换。

## A1：移除布尔 BEV 掩码的冗余清洗

目标为 `BEVFormerEncoder.point_sampling`。基线使用比较与逻辑运算构造 `torch.bool` 类型的 `bev_mask`，然后按 PyTorch 版本调用 `torch.nan_to_num` 或 NumPy 兜底路径。

布尔张量不可能包含 NaN 或 Inf，因此两条清洗分支对掩码数值都是恒等变换。优化实现删除整段清洗，保留 `force_fp32` 装饰器、投影计算、可见性判断和返回布局。

现代 PyTorch 的基线已走 Torch 路径，A1 的收益是减少冗余算子及其可能的分配、复制；旧版 NumPy 分支还涉及设备往返。具体代价依赖版本与后端，应以目标环境 profiler 为准。非有限投影坐标的处理行为保持基线语义，A1 不清洗返回坐标。

## A2：移除正样本归一化的 CPU 标量读取

三个目标为 `MapTRv2Head.loss_single`、`MapTRHead.loss_single` 和 `BEVFormerHead.loss_single`。将正样本归一化因子的 `.item()` 替换为 `.reshape(())`：

```python
num_total_pos = loss_cls.new_tensor([num_total_pos])
num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).reshape(())
```

输入与归约结果原本为单元素一维张量，`reshape(())` 将其变为设备上的零维张量，避免 `.item()` 读取 CPU，同时保留 loss 输出形状。分布式归约、最小值截断及原始 loss 的版本分支保持原样。

公开 `loss` 入口及静态 loss 路径保留 `force_fp32` 精度契约。FP16 预测应通过公开入口验证，实际 `loss_single` 通常接收 FP32 输入。

## 安装与回退

实现位于 `sync_elimination.py`，由 catalog 安装方法替换，无需修改上游模型源码。两份配置中的 `enabled` 控制整个组；修改开关后应重启训练进程。

组不再设置额外的 PyTorch >= 1.8 / `nan_to_num` 可调用性检查。引擎继续执行目标解析、签名及基线源码身份检查，配置中的 source/AST trust 证据仍对应原始方法。取消额外检查不代表所有旧版依赖组合均已验证。

## 验证

专项测试覆盖基线两条 A1 版本分支、有限与非有限坐标、布尔掩码、形状、梯度及 FP16 入口；检查 A1 不调用 Torch/NumPy 清洗。安装测试覆盖四个目标分别 trust 失败、关闭开关以及中途失败回退。A2 保留真实静态 assigner 与 FP16 公开 loss 入口联合验证。

```bash
bash scripts/test_maptrv2.sh -q -rs
```

A1 的真实调用验证使用 `projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py`，其 encoder 为 `BEVFormerEncoder`。默认 `maptrv2_nusc_r50_24ep.py` 使用 `LSSTransform`，无法用于测量 A1 收益。

性能验证应先确认 `point_sampling` 被调用，固定输入和其他优化项，预热后比较方法耗时与 profiler 算子、分配及复制记录。方法级结果不能直接视为训练吞吐收益；完整训练收益须另行测量，不设波动敏感的性能单测门槛。

### 实机验证记录

在 PyTorch 2.7.1 / HCU 环境执行完整脚本：210 个测试及 139 个子测试通过，无跳过。真实 BEVFormer encoder 前向确认调用替换后的 `point_sampling`，输出与基线一致；profiler 确认优化路径无 `aten::nan_to_num`。

方法级测量使用上述 BEVFormer 配置的 200×100 网格、4 个高度采样点、6 个相机、batch=1，投影元数据为合成输入。每个实现预热 10 次，再交替测量 5 组，每组 20 次调用，组前后同步设备。基线与优化后的墙钟耗时中位数分别约为 2.004 ms 和 1.998 ms。此微小差异仅为该环境的单次测量记录，不作为稳定加速承诺。

基线一次清洗在 profiler 中呈现两条嵌套 `aten::nan_to_num` 事件，聚合设备内存为 480,256 字节；优化路径两者消失。这不代表两次独立清洗。完整训练吞吐尚未测量。
