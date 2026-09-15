# upsample_bilinear_2d

## 接口原型

```python
# 带自动求导的 Python 封装
turbo_physai.interpolate(
    input, size=None, scale_factor=None, mode="bilinear",
    align_corners=None, recompute_scale_factor=None, antialias=False
) -> Tensor

# 原生前向 / 反向
turbo_physai.ops.upsample_bilinear_2d_forward(
    input, output_size, align_corners, scale_factors
) -> Tensor
turbo_physai.ops.upsample_bilinear_2d_backward(
    grad_output, output_size, input_size, align_corners, scale_factors
) -> Tensor
```

接口位置：`turbo_physai`（顶层导出）、`turbo_physai.ops`（原生）。实现见 `turbo_physai/operators/upsample_bilinear_2d.py`。

## 功能描述

2D 双线性上采样，等价于 `torch.nn.functional.interpolate` 的 `mode="bilinear"` 分支。BEVFormer、Sparse4D 等模型用它上采样 BEV 特征。

## 参数说明

Python 封装：

- `input(Tensor)`：shape `[N, C, H_in, W_in]`。
- `size(int | Tuple[int, int] | None)`：输出尺寸 `(H_out, W_out)`；也可传入标量，由封装扩展为两个空间维度。
- `scale_factor(float | Tuple[float, float] | None)`：缩放因子。
- `mode(str)`：仅支持 `"bilinear"`。
- `align_corners(bool | None)`：为 `True` 时报 `ValueError`；`None` 按 `False` 处理。
- `recompute_scale_factor(bool | None)`：使用 `scale_factor` 时，若为 `True`，先根据缩放因子确定整数输出尺寸，再依据该输出尺寸进行插值；传入显式 `size` 时不能设为 `True`。
- `antialias(bool)`：为 `True` 时报 `ValueError`。

原生接口：

- `output_size(List[int] | None)`：长度 2，指定输出尺寸；与 `scale_factors` 二选一。
- `input_size(List[int])`：反向需知输入尺寸，格式 `[N, C, H_in, W_in]`。
- `align_corners(bool)`。
- `scale_factors(List[float] | None)`：长度 2，与 `output_size` 二选一。

## 返回值

- 前向：上采样结果，shape `[N, C, H_out, W_out]`。
- 反向：输入梯度，shape 与 `input` 一致。

## 约束说明

- Python 封装：`mode` 必须为 `"bilinear"`；`align_corners=True` / `antialias=True` / 非 4D 输入均报错；`size` 与 `scale_factor` 必须传且只能传一个。
- 内核源码注释提示"目前只针对特定 size 优化，其他 size 可能会 coredump"，使用非常规尺寸需注意验证。

## 调用示例

```python
import torch
from turbo_physai import interpolate
input = torch.randn(
    2,
    64,
    32,
    32,
    device="cuda",
    dtype=torch.float32,
    requires_grad=True,
)
out = interpolate(
    input,
    size=(64, 64),
    mode="bilinear",
    align_corners=False,
)
out.sum().backward()
torch.cuda.synchronize()
print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"input.grad shape: {input.grad.shape}, "
    f"input.grad dtype: {input.grad.dtype}"
)
```

一次运行的输出示例：

```text
output shape: torch.Size([2, 64, 64, 64]), output dtype: torch.float32, input.grad shape: torch.Size([2, 64, 32, 32]), input.grad dtype: torch.float32
```

## 参考

- 源码：`kernel/upsample_bilinear_2d/UpSampleBilinear2d.cu`
- Python 封装：`turbo_physai/operators/upsample_bilinear_2d.py`
- 算子测试：`test/test_upsample_bilinear_2d.py`
- 返回[算子 API 清单](../README.md)
