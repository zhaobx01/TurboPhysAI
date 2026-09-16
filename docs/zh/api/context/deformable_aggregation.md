# deformable_aggregation

## 接口原型

```python
turbo_physai.ops.deformable_aggregation_forward(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights
) -> Tensor
turbo_physai.ops.deformable_aggregation_backward(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights
) -> None
```

接口位置：`turbo_physai.ops`。该接口直接调用编译后的原生扩展，不自动完成 dtype、连续性和自动求导处理。

## 功能描述

多相机多尺度可变形聚合：在 `sampling_location` 指定位置对多相机、多尺度特征做双线性采样，再按 `weights` 加权求和输出每个 anchor 的特征。Sparse4D 用它替代可变形注意力。

## 参数说明

- `feature_maps / mc_ms_feat(Tensor)`：float32，shape `[B, num_feat, C]`，`num_feat` 为所有相机、所有尺度特征点展平后的总数。
- `spatial_shape(Tensor)`：int32，shape `[cam, scale, 2]`，最后一维为 `(H, W)`。
- `scale_start_index(Tensor)`：int32，shape `[cam, scale]`，每个相机尺度在 `num_feat` 中的起始下标。
- `sampling_location(Tensor)`：float32，shape `[B, anchor, pts, cam, 2]`，使用约 `[0, 1]` 范围的归一化坐标 `(x, y)`；该约定不同于 `grid_sample` 使用的 `[-1, 1]` 坐标。
- `weights(Tensor)`：float32，shape `[B, anchor, pts, cam, scale, numGroups]`；输出通道按 group 划分，需满足 `C % numGroups == 0`。
- `grad_output(Tensor)`：反向上游梯度，float32，shape `[B, anchor, C]`。
- `grad_mc_ms_feat, grad_sampling_location, grad_weights(Tensor)`：反向输出缓冲区，由调用方预先分配并置零，shape 与对应前向输入一致。

## 返回值

- 前向：float32，shape `[B, anchor, C]`。
- 反向：无返回值，结果写入 `grad_mc_ms_feat`、`grad_sampling_location`、`grad_weights`。

## 约束说明

- 原生扩展接口不负责参数整理；调用者需自行保证 dtype、设备和连续性。
- 反向不对 `spatial_shape` 与 `scale_start_index` 求梯度。

## 调用示例

```python
import torch
from turbo_physai import ops
B, C, numGroups = 1, 32, 8
anchor, pts, cam, scale, H, W = 10, 10, 1, 1, 32, 88
num_feat = cam * scale * H * W
feature_maps = torch.rand(B, num_feat, C, device="cuda")
spatial_shape = torch.tensor([[[H, W]] * scale] * cam, dtype=torch.int32, device="cuda")
scale_start_index = torch.arange(cam * scale, dtype=torch.int32, device="cuda").view(cam, scale) * (H * W)
sampling_location = torch.rand(B, anchor, pts, cam, 2, device="cuda")
weights = torch.rand(B, anchor, pts, cam, scale, numGroups, device="cuda")
out = ops.deformable_aggregation_forward(
    feature_maps, spatial_shape, scale_start_index, sampling_location, weights
)
grad_feature_maps = torch.zeros_like(feature_maps)
grad_sampling_location = torch.zeros_like(sampling_location)
grad_weights = torch.zeros_like(weights)
ops.deformable_aggregation_backward(
    feature_maps, spatial_shape, scale_start_index, sampling_location, weights,
    torch.ones_like(out), grad_feature_maps, grad_sampling_location, grad_weights,
)
torch.cuda.synchronize()
print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"feature_maps grad shape: {grad_feature_maps.shape}, "
    f"feature_maps grad dtype: {grad_feature_maps.dtype}, "
    f"sampling_location grad shape: {grad_sampling_location.shape}, "
    f"sampling_location grad dtype: {grad_sampling_location.dtype}, "
    f"weights grad shape: {grad_weights.shape}, weights grad dtype: {grad_weights.dtype}"
)
```

一次运行的输出示例：
```text
output shape: torch.Size([1, 10, 32]), output dtype: torch.float32, feature_maps grad shape: torch.Size([1, 2816, 32]), feature_maps grad dtype: torch.float32, sampling_location grad shape: torch.Size([1, 10, 10, 1, 2]), sampling_location grad dtype: torch.float32, weights grad shape: torch.Size([1, 10, 10, 1, 1, 8]), weights grad dtype: torch.float32
```

## 参考

- 源码：`kernel/deformable_aggregation/DeformableAggregation.cu`
- 可选的高层兼容层：`turbo_physai/operators/deformable_aggregation.py`
- 算子测试：`test/test_deformable_aggregation.py`
- 返回[算子 API 清单](../README.md)
