# grid_sample

## 接口原型

```python
# 带自动求导的 Python 封装
turbo_physai.grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=None
) -> Tensor

# 原生前向 / 反向
turbo_physai.ops.grid_sample_forward(
    input, grid, interpolation_mode, padding_mode, align_corners
) -> Tensor
turbo_physai.ops.grid_sample_backward(
    grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask
) -> Tuple[Tensor, Tensor]
```

接口位置：`turbo_physai`（顶层导出）、`turbo_physai.ops`（原生）。实现见 `turbo_physai/operators/grid_sample.py`。

## 功能描述

2D 双线性网格采样：按 `grid` 给出的归一化坐标在 `input` 上做双线性插值，等价于 `torch.nn.functional.grid_sample` 的 `bilinear` 分支。

## 参数说明

Python 封装：

- `input(Tensor)`：shape `[N, C, H_in, W_in]`。
- `grid(Tensor)`：shape `[N, H_out, W_out, 2]`，最后一维为归一化坐标 `(x, y)`。
- `mode(str)`：仅支持 `"bilinear"`。
- `padding_mode(str)`：`"zeros"` / `"border"` / `"reflection"`。
- `align_corners(bool | None)`：`None` 时按 `False` 处理并 `warnings.warn`。

原生接口的枚举参数：

- `interpolation_mode(int)`：`0=bilinear`、`1=nearest`、`2=bicubic`；封装内固定传 `0`。
- `padding_mode(int)`：`0=zeros`、`1=border`、`2=reflection`。
- `output_mask(List[bool])`：长度 2，分别控制是否输出 `input` 与 `grid` 的梯度；Python 封装始终传 `[True, True]`。

## 返回值

- 前向：采样结果，shape `[N, C, H_out, W_out]`。
- 反向：`(grad_input, grad_grid)`。

## 约束说明

- Python 封装仅接受 4 维 `input` 与 `grid`、`mode="bilinear"`、`padding_mode ∈ {"zeros", "border", "reflection"}`；不符合直接 `ValueError`。
- 如需只求 `input` 或 `grid` 单方梯度，直接调原生 `grid_sample_backward` 并调整 `output_mask`。

## 调用示例

```python
import torch
from turbo_physai import grid_sample

input = torch.randn(
    2,
    32,
    64,
    64,
    device="cuda",
    dtype=torch.float32,
    requires_grad=True,
)

grid = (
    torch.rand(
        2,
        128,
        128,
        2,
        device="cuda",
        dtype=torch.float32,
    )
    * 2
    - 1
).requires_grad_()

# 前向传播
out = grid_sample(
    input,
    grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
)

# 反向传播
out.sum().backward()

torch.cuda.synchronize()

print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"input.grad shape: {input.grad.shape}, "
    f"input.grad dtype: {input.grad.dtype}, "
    f"grid.grad shape: {grid.grad.shape}, "
    f"grid.grad dtype: {grid.grad.dtype}"
)
```

一次运行的输出示例：

```text
output shape: torch.Size([2, 32, 128, 128]), output dtype: torch.float32, input.grad shape: torch.Size([2, 32, 64, 64]), input.grad dtype: torch.float32, grid.grad shape: torch.Size([2, 128, 128, 2]), grid.grad dtype: torch.float32
```

## 参考

- 源码：`kernel/grid_sample/GridSampler.cu`
- Python 封装：`turbo_physai/operators/grid_sample.py`
- 算子测试：`test/test_grid_sample.py`
- 返回[算子 API 清单](../README.md)