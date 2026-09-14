# get_indice_pairs

## 接口原型

```python
# Python 封装：自动推导输出形状并广播标量参数
turbo_physai.optimizations.common.mmdet3d.sparse_conv.get_indice_pairs(
    indices, batch_size, spatial_shape, ksize=3, stride=1, padding=0,
    dilation=1, out_padding=0, subm=False, transpose=False, grid=None
) -> Tuple[Tensor, Tensor, Tensor]

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

接口位置：`turbo_physai.ops`。Python 封装见 `turbo_physai/optimizations/common/mmdet3d/sparse_conv.py`。

## 功能描述

生成稀疏卷积所需的索引对：对每个卷积核偏移 `k`，计算"哪些输入激活与哪些输出激活相连"，输出：输出坐标表、索引对表、以及每个核偏移的有效对数量。是 `indice_conv` / `indice_maxpool` 的前置步骤。

## 参数说明

Python 封装：

- `indices(Tensor)`：int32，shape `[num_points, NDim + 1]`，第一列为 batch 索引。
- `batch_size(int)`：batch 大小。
- `spatial_shape(list[int])`：输入空间形状，长度 `NDim`。
- `ksize / stride / padding / dilation / out_padding`：卷积核、步长、填充、膨胀、输出填充，标量或长度 `NDim` 的列表。
- `subm(bool)`：子流形卷积；为 `True` 时输出形状等于 `spatial_shape`。
- `transpose(bool)`：转置卷积；为 `True` 时按反卷积公式计算输出形状。
- `grid(Tensor | None)`：预计算的输出网格；为 `None` 时调用普通版本，否则走 `get_indice_pairs_grid_*`。

原生接口：

- `out_spatial_shape(list[int])`：输出空间形状；非 subm 时按公式提前算出。
- `subm / transpose(int)`：`0` / `1`。
- `grid(Tensor)`：int32，长度 `batch_size * prod(out_spatial_shape)`，元素为输出激活下标或 `-1`。

## 返回值

- `out_inds(Tensor)`：int32，shape `[numActOut, NDim + 1]`；`subm=True` 时直接原样返回输入 `indices`，因此空输入返回 shape `[0, NDim + 1]` 的空坐标表；非 `subm` 模式下，`numActOut` 由实际生成的输出坐标决定，可能不同于输入激活数，空输入行为应以当前底层实现为准。
- `indice_pairs(Tensor)`：int32，shape `[kernelVolume, 2, numActIn]`，初值 `-1`；`[k, 0]` 为输入下标、`[k, 1]` 为输出下标。
- `indice_num(Tensor)`：int32，shape `[kernelVolume]`，每个核偏移的有效索引对数量。

## 约束说明

- 同时提供 CPU 与 HCU（ROCm/HIP）实现。
- `NDim` 必须等于 `indices.size(1) - 1`；所有尺寸列表长度必须为 `NDim`。
- `kernelVolume = prod(kernel_size)` 必须 `<= 4096`。
- Python 封装约束：`stride` 与 `dilation` 不能同时大于 1；只支持 `NDim ∈ {2, 3, 4}`（grid 变体仅 `{2, 3}`）。
- `subm=True` 时内核内部强制 `stride=1`、`padding=kernel_size // 2`，外部传入的这两项会被覆盖。
- 环境变量 `MMDET3D_SPCONV_CANONICAL_INDICE=1` 会对每个核偏移的索引对做规范化排序，便于结果复现。

## 调用示例

```python
import torch
from turbo_physai.optimizations.common.mmdet3d.sparse_conv import (
    get_indice_pairs,
)

# 4 个稀疏激活点，坐标列为 (batch, z, y, x)
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

out_inds, indice_pairs, indice_num = get_indice_pairs(
    indices,
    batch_size=1,
    spatial_shape=[5, 5, 5],
    ksize=3,
    subm=True,
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
- Python 封装：`turbo_physai/optimizations/common/mmdet3d/sparse_conv.py`
- 相关接口：[indice_conv](./indice_conv.md)、[indice_maxpool](./indice_maxpool.md)
- 返回[算子 API 清单](../README.md)