# dynamic_point_to_voxel

## 接口原型

```python
from turbo_physai import ops

ops.dynamic_point_to_voxel_forward(
    feats, coors, reduce_type
) -> (reduced_feats, out_coors, coors_map, reduce_count)

ops.dynamic_point_to_voxel_backward(
    grad_feats,
    grad_reduced_feats,
    feats,
    reduced_feats,
    coors_idx,
    reduce_count,
    reduce_type,
) -> None
```

接口位置：`turbo_physai.ops`，语义等价于 MMDetection3D 的 `DynamicScatter`。

## 功能描述

点特征到体素的归约：把同一体素内所有点的特征按 `reduce_type` 归约成一个体素特征，并给出点与体素的双向映射。BEVFusion、Sparse4D 等模型用它完成"点云特征 → 稀疏体素特征"。

- 前向：`feats + coors → 归约特征 + 去重体素坐标 + 映射表 + 计数`。
- 反向：按 `reduce_type` 把体素梯度分摊回每个点。

## 参数说明

前向：

- `feats(Tensor)`：float32，shape `[N, C]`。
- `coors(Tensor)`：int32，shape `[N, 3]`；整行为负数的点视为无效并被跳过。
- `reduce_type(str)`：`"max"`、`"sum"`、`"mean"` 之一。

反向：

- `grad_feats(Tensor)`：**输出缓冲区**，函数会先整体置零。
- `grad_reduced_feats(Tensor)`：体素特征上游梯度，shape 与 `reduced_feats` 一致。
- `feats, reduced_feats(Tensor)`：前向输入与输出，反向据此定位 `max` 的取值位置。
- `coors_idx(Tensor)`：前向返回的 `coors_map`（注意不是 `out_coors`）。
- `reduce_count(Tensor)`：前向返回的 `reduce_count`。
- `reduce_type(str)`：与前向保持一致。

## 返回值

前向返回 4 个张量：

- `reduced_feats(Tensor)`：float32，shape `[num_voxels, C]`。
- `out_coors(Tensor)`：int32，shape `[num_voxels, 3]`。
- `coors_map(Tensor)`：int32，shape `[N]`，每个点映射到的体素下标，无效点为 `-1`。
- `reduce_count(Tensor)`：int32，shape `[num_voxels]`。

反向无返回值，结果写入 `grad_feats`。

## 约束说明

- 仅支持 HCU（ROCm/HIP）设备，CPU 调用直接报错 `do not support cpu yet`。
- `N == 0` 时前向返回 `(feats.clone(), coors.clone(), empty_coors_map, empty_reduce_count)`；后两个张量为空且为 int32。
- `"max"` 的初值为 `-inf`；反向将梯度回传给选中的最大值点；如果同一体素、同一通道存在并列最大值，当前实现选择输入索引较小的点。`"mean"` 在归约后除以 `reduce_count`。

## 调用示例

```python
import torch
from turbo_physai import ops
feats = torch.randn(
    20000,
    32,
    device="cuda",
    dtype=torch.float32,
)
coors = torch.randint(
    0,
    100,
    (20000, 3),
    device="cuda",
    dtype=torch.int32,
)
reduced, out_coors, coors_map, reduce_count = (
    ops.dynamic_point_to_voxel_forward(
        feats,
        coors,
        "max",
    )
)
torch.cuda.synchronize()
grad_feats = torch.empty_like(feats)
ops.dynamic_point_to_voxel_backward(
    grad_feats,
    torch.ones_like(reduced),
    feats,
    reduced,
    coors_map,
    reduce_count,
    "max",
)
torch.cuda.synchronize()
print(
    f"reduced shape: {reduced.shape}, reduced dtype: {reduced.dtype}, "
    f"out_coors shape: {out_coors.shape}, out_coors dtype: {out_coors.dtype}, "
    f"coors_map shape: {coors_map.shape}, coors_map dtype: {coors_map.dtype}, "
    f"reduce_count shape: {reduce_count.shape}, reduce_count dtype: {reduce_count.dtype}, "
    f"grad_feats shape: {grad_feats.shape}, grad_feats dtype: {grad_feats.dtype}"
)
```

一次运行的输出示例：
```text
reduced shape: torch.Size([19815, 32]), reduced dtype: torch.float32, out_coors shape: torch.Size([19815, 3]), out_coors dtype: torch.int32, coors_map shape: torch.Size([20000]), coors_map dtype: torch.int32, reduce_count shape: torch.Size([19815]), reduce_count dtype: torch.int32, grad_feats shape: torch.Size([20000, 32]), grad_feats dtype: torch.float32
```

## 参考

- 源码：`kernel/voxelization/src/voxelization.h`、`kernel/voxelization/src/scatter_points_cuda.cu`
- 相关接口：[hard_voxelize](./hard_voxelize.md)、[dynamic_voxelize](./dynamic_voxelize.md)
- 返回[算子 API 清单](../README.md)
