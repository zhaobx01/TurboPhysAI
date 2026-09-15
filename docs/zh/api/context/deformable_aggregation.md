# deformable_aggregation

## 接口原型

```python
# 带自动求导的 Python 封装
turbo_physai.deformable_aggregation_function(
    feature_maps, spatial_shape, scale_start_index, sampling_location, weights
) -> Tensor

# 原生前向 / 反向
turbo_physai.ops.deformable_aggregation_forward(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights
) -> Tensor
turbo_physai.ops.deformable_aggregation_backward(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights
) -> None
```

接口位置：`turbo_physai`（顶层导出）、`turbo_physai.ops`（原生）。实现见 `turbo_physai/operators/deformable_aggregation.py`。

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

Python 封装 `turbo_physai.deformable_aggregation_function`：

- 会将参与计算的浮点输入转换为 contiguous 的 float32 张量，将 `spatial_shape` 和 `scale_start_index` 转换为 contiguous 的 int32 张量。
- 调用者通常不需要手动执行上述转换，但仍需提供符合 shape 和设备要求的输入。

原生扩展接口 `turbo_physai.ops.deformable_aggregation_forward/backward`：

- 不负责 Python 封装层的参数整理；调用者需自行保证 dtype、设备和连续性。
- 反向不对 `spatial_shape` 与 `scale_start_index` 求梯度。

## 调用示例

```python
import torch
from turbo_physai import deformable_aggregation_function
B, C, numGroups = 1, 32, 8
anchor, pts, cam, scale, H, W = 10, 10, 1, 1, 32, 88
num_feat = cam * scale * H * W
feature_maps = torch.rand(B, num_feat, C, device="cuda", requires_grad=True)
spatial_shape = torch.tensor([[[H, W]] * scale] * cam, dtype=torch.int32, device="cuda")
scale_start_index = torch.arange(cam * scale, dtype=torch.int32, device="cuda").view(cam, scale) * (H * W)
sampling_location = torch.rand(B, anchor, pts, cam, 2, device="cuda", requires_grad=True)
weights = torch.rand(B, anchor, pts, cam, scale, numGroups, device="cuda", requires_grad=True)
out = deformable_aggregation_function(
    feature_maps, spatial_shape, scale_start_index, sampling_location, weights
)
out.sum().backward()
torch.cuda.synchronize()
print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"feature_maps.grad shape: {feature_maps.grad.shape}, "
    f"feature_maps.grad dtype: {feature_maps.grad.dtype}, "
    f"sampling_location.grad shape: {sampling_location.grad.shape}, "
    f"sampling_location.grad dtype: {sampling_location.grad.dtype}, "
    f"weights.grad shape: {weights.grad.shape}, "
    f"weights.grad dtype: {weights.grad.dtype}"
)
```

一次运行的输出示例：
```text
output shape: torch.Size([1, 10, 32]), output dtype: torch.float32, feature_maps.grad shape: torch.Size([1, 2816, 32]), feature_maps.grad dtype: torch.float32, sampling_location.grad shape: torch.Size([1, 10, 10, 1, 2]), sampling_location.grad dtype: torch.float32, weights.grad shape: torch.Size([1, 10, 10, 1, 1, 8]), weights.grad dtype: torch.float32
```

## 参考

- 源码：`kernel/deformable_aggregation/DeformableAggregation.cu`
- Python 封装：`turbo_physai/operators/deformable_aggregation.py`
- 算子测试：`test/test_deformable_aggregation.py`
- 返回[算子 API 清单](../README.md)
