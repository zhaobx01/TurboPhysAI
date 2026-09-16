# grid_sample

## 接口原型

```python
turbo_physai.ops.grid_sample_forward(
    input, grid, interpolation_mode, padding_mode, align_corners
) -> Tensor
turbo_physai.ops.grid_sample_backward(
    grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask
) -> Tuple[Tensor, Tensor]
```

接口位置：`turbo_physai.ops`。本文接口和示例均直接调用编译后的原生扩展。

## 功能描述

2D 双线性网格采样：按 `grid` 给出的归一化坐标在 `input` 上做双线性插值，等价于 `torch.nn.functional.grid_sample` 的 `bilinear` 分支。

## 参数说明

前向接口：

- `input(Tensor)`：shape `[N, C, H_in, W_in]`。
- `grid(Tensor)`：shape `[N, H_out, W_out, 2]`，最后一维为归一化坐标 `(x, y)`。
- `interpolation_mode(int)`：接口保留 `0=bilinear`、`1=nearest`、`2=bicubic` 的枚举值；当前 kernel 仅实现 `0`（bilinear）。
- `padding_mode(int)`：`0=zeros`、`1=border`、`2=reflection`。
- `align_corners(bool)`。

反向接口参数：

- `output_mask(List[bool])`：长度 2，分别控制是否输出 `input` 与 `grid` 的梯度。

## 返回值

- 前向：采样结果，shape `[N, C, H_out, W_out]`。
- 反向：`(grad_input, grad_grid)`。

## 约束说明

- 当前 kernel 仅实现 2D bilinear；原生接口不负责字符串模式转换、参数整理或自动求导，调用方需自行保证输入形状、dtype、设备和连续性。
- 如需只求 `input` 或 `grid` 单方梯度，可调整 `output_mask`。

## 调用示例

```python
import torch
from turbo_physai import ops
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
out = ops.grid_sample_forward(input, grid, 0, 0, False)
grad_input, grad_grid = ops.grid_sample_backward(
    torch.ones_like(out), input, grid, 0, 0, False, [True, True]
)
torch.cuda.synchronize()
print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"input grad shape: {grad_input.shape}, input grad dtype: {grad_input.dtype}, "
    f"grid grad shape: {grad_grid.shape}, grid grad dtype: {grad_grid.dtype}"
)
```

一次运行的输出示例：

```text
output shape: torch.Size([2, 32, 128, 128]), output dtype: torch.float32, input.grad shape: torch.Size([2, 32, 64, 64]), input.grad dtype: torch.float32, grid.grad shape: torch.Size([2, 128, 128, 2]), grid.grad dtype: torch.float32
```

## 参考

- 源码：`kernel/grid_sample/GridSampler.cu`
- 可选的高层兼容层：`turbo_physai/operators/grid_sample.py`
- 算子测试：`test/test_grid_sample.py`
- 返回[算子 API 清单](../README.md)
