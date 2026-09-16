# get_indice_pairs

## 接口原型

```python
# 原生接口（按维度选择 2d / 3d / 4d；grid 变体仅 2d / 3d）
turbo_physai.ops.get_indice_pairs_3d(
    indices, batch_size, out_spatial_shape, spatial_shape,
    kernel_size, stride, padding, dilation, out_padding, subm, transpose
) -> Tuple[Tensor, Tensor, Tensor]

turbo_physai.ops.get_indice_pairs_grid_3d(
    indices, grid, batch_size, out_spatial_shape, spatial_shape,
    kernel_size, stride, padding, dilation, out_padding, subm, transpose
) -> Tuple[Tensor, Tensor, Tensor]
```

接口位置：`turbo_physai.ops`。下面的调用直接使用原生扩展；参数广播和输出形状需要由调用方自行准备。

## 功能描述

生成稀疏卷积所需的索引对：对每个卷积核偏移 `k`，计算"哪些输入激活与哪些输出激活相连"，输出：输出坐标表、索引对表、以及每个核偏移的有效对数量。是 `indice_conv` / `indice_maxpool` 的前置步骤。

## 参数说明

- `indices(Tensor)`：int32，shape `[num_points, NDim + 1]`，第一列为 batch 索引。
- `batch_size(int)`：batch 大小。
- `out_spatial_shape(list[int])`：输出空间形状，由调用方提前计算。
- `spatial_shape(list[int])`：输入空间形状，长度 `NDim`。
- `kernel_size / stride / padding / dilation / out_padding`：长度均为 `NDim` 的列表。
- `subm / transpose(int)`：`0` / `1`；这是 native binding 的整数开关，不是 Python `bool` 参数。
- `grid(Tensor)`：int32，长度 `batch_size * prod(out_spatial_shape)`，元素为输出激活下标或 `-1`。

## 返回值

- `out_inds(Tensor)`：int32，shape `[numActOut, NDim + 1]`；`subm=True` 时直接原样返回输入 `indices`，因此空输入返回 shape `[0, NDim + 1]` 的空坐标表；非 `subm` 模式下，`numActOut` 由实际生成的输出坐标决定，可能不同于输入激活数，空输入行为应以当前底层实现为准。
- `indice_pairs(Tensor)`：int32，shape `[kernelVolume, 2, numActIn]`，初值 `-1`；`[k, 0]` 为输入下标、`[k, 1]` 为输出下标。
- `indice_num(Tensor)`：int32，shape `[kernelVolume]`，每个核偏移的有效索引对数量。

## 约束说明

- 同时提供 CPU 与 HCU（ROCm/HIP）实现。
- `NDim` 必须等于 `indices.size(1) - 1`；所有尺寸列表长度必须为 `NDim`。
- `kernelVolume = prod(kernel_size)` 必须 `<= 4096`。
- 原生接口只支持 `NDim ∈ {2, 3, 4}`（grid 变体仅 `{2, 3}`）；调用方需自行保证各参数列表长度和输出形状正确。
- `subm=True` 时内核内部强制 `stride=1`、`padding=kernel_size // 2`，外部传入的这两项会被覆盖。
- 环境变量 `MMDET3D_SPCONV_CANONICAL_INDICE=1` 会对每个核偏移的索引对做规范化排序，便于结果复现。

## 调用示例

```python
import torch
from turbo_physai import ops
indices = torch.tensor(
    [
        [0, 1, 2, 3],
        [0, 2, 2, 3],
        [0, 1, 3, 3],
        [0, 2, 2, 3],
    ],
    dtype=torch.int32,
    device="cuda",
)
out_inds, indice_pairs, indice_num = ops.get_indice_pairs_3d(
    indices,
    1,
    [5, 5, 5],
    [5, 5, 5],
    [3, 3, 3],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    1,
    0,
)
torch.cuda.synchronize()
print(
    f"indices shape: {indices.shape}, indices dtype: {indices.dtype}, "
    f"out_inds shape: {out_inds.shape}, out_inds dtype: {out_inds.dtype}, "
    f"indice_pairs shape: {indice_pairs.shape}, "
    f"indice_pairs dtype: {indice_pairs.dtype}, "
    f"indice_num shape: {indice_num.shape}, "
    f"indice_num dtype: {indice_num.dtype}"
)
```

一次运行的输出示例：

```text
indices shape: torch.Size([4, 4]), indices dtype: torch.int32, out_inds shape: torch.Size([4, 4]), out_inds dtype: torch.int32, indice_pairs shape: torch.Size([27, 2, 4]), indice_pairs dtype: torch.int32, indice_num shape: torch.Size([27]), indice_num dtype: torch.int32
```

## 参考

- 源码：`kernel/sparse_conv/include/spconv/spconv_ops.h`、`kernel/sparse_conv/src/indice_cpu.cc`、`kernel/sparse_conv/src/indice_cuda.cu`
- 可选的模型兼容辅助层：`turbo_physai/operators/sparse_conv.py`
- 相关接口：[indice_conv](./indice_conv.md)、[indice_maxpool](./indice_maxpool.md)
- 返回[算子 API 清单](../README.md)
