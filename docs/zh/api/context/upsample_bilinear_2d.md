# upsample_bilinear_2d

## 接口原型

```python
turbo_physai.ops.upsample_bilinear_2d_forward(
    input, output_size, align_corners, scale_factors
) -> Tensor
turbo_physai.ops.upsample_bilinear_2d_backward(
    grad_output, output_size, input_size, align_corners, scale_factors
) -> Tensor
```

接口位置：`turbo_physai.ops`。本文接口和示例均直接调用编译后的原生扩展。

## 功能描述

2D 双线性上采样，等价于 `torch.nn.functional.interpolate` 的 `mode="bilinear"` 分支。BEVFormer、Sparse4D 等模型用它上采样 BEV 特征。

## 参数说明

- `input(Tensor)`：shape `[N, C, H_in, W_in]`。

- `output_size(List[int] | None)`：长度 2，指定输出尺寸；与 `scale_factors` 二选一。
- `input_size(List[int])`：反向需知输入尺寸，格式 `[N, C, H_in, W_in]`。
- `align_corners(bool)`；参数为接口兼容而保留，当前 kernel 按固定的 `align_corners=False` 坐标规则计算，调用示例传入 `False`。
- `scale_factors(List[float] | None)`：长度 2，与 `output_size` 二选一。

## 返回值

- 前向：上采样结果，shape `[N, C, H_out, W_out]`。
- 反向：输入梯度，shape 与 `input` 一致。

## 约束说明

- 原生接口只实现 bilinear 4D 输入；`output_size` 与 `scale_factors` 由 PyTorch native 输出尺寸计算逻辑处理，调用方应二选一；当前实现使用固定的 half-pixel 坐标规则，`align_corners` 参数不会改变实际计算。
- 内核源码注释提示"目前只针对特定 size 优化，其他 size 可能会 coredump"，使用非常规尺寸需注意验证。

## 调用示例

```python
import torch
from turbo_physai import ops
input = torch.randn(
    2,
    64,
    32,
    32,
    device="cuda",
    dtype=torch.float32,
    requires_grad=True,
)
out = ops.upsample_bilinear_2d_forward(input, [64, 64], False, None)
grad_input = ops.upsample_bilinear_2d_backward(
    torch.ones_like(out), [64, 64], list(input.shape), False, None
)
torch.cuda.synchronize()
print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"input grad shape: {grad_input.shape}, input grad dtype: {grad_input.dtype}"
)
```

一次运行的输出示例：

```text
output shape: torch.Size([2, 64, 64, 64]), output dtype: torch.float32, input.grad shape: torch.Size([2, 64, 32, 32]), input.grad dtype: torch.float32
```

## 参考

- 源码：`kernel/upsample_bilinear_2d/UpSampleBilinear2d.cu`
- 可选的高层兼容层：`turbo_physai/operators/upsample_bilinear_2d.py`
- 算子测试：`test/test_upsample_bilinear_2d.py`
- 返回[算子 API 清单](../README.md)
