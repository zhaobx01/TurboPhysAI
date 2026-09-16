# indice_conv

## 接口原型

```python
turbo_physai.ops.indice_conv_fp32(
    features, filters, indice_pairs, indice_num, num_act_out, inverse, sub_m
) -> Tensor
turbo_physai.ops.indice_conv_backward_fp32(
    features, filters, out_grad, indice_pairs, indice_num, inverse, sub_m
) -> Tuple[Tensor, Tensor]

# 融合偏置的版本
turbo_physai.ops.fused_indice_conv_fp32(
    features, filters, bias, indice_pairs, indice_num, num_act_out, inverse, sub_m
) -> Tensor

# float16 版本：indice_conv_half / indice_conv_backward_half / fused_indice_conv_half
```

接口位置：`turbo_physai.ops`。以下示例直接调用 native extension；可选的模型兼容辅助层见 `turbo_physai/operators/sparse_conv.py`。native binding 见 `kernel/sparse_conv/src/all.cc`。

## 功能描述

稀疏卷积：按 `indice_pairs` 给出的连接关系做 gather → 矩阵乘 → scatter-add，只对有效激活位置计算。与 `get_indice_pairs` 配套：前者生成索引对，本接口消费索引对完成卷积。

`fused_indice_conv_*` 是带偏置的变体，输出以 `bias` 为初值再累加各核偏移的贡献。

## 参数说明

- `features(Tensor)`：输入激活，shape `[numActIn, numInPlanes]`。
- `filters(Tensor)`：卷积核权重，shape `[k_0, ..., k_{ndim-1}, numInPlanes, numOutPlanes]`；内核会 view 成 `[kernelVolume, numInPlanes, numOutPlanes]`。
- `bias(Tensor)`：可广播到 `[numActOut, numOutPlanes]`，通常 `[numOutPlanes]`。仅 `fused_indice_conv_*` 需要。
- `indice_pairs, indice_num(Tensor)`：`get_indice_pairs` 的输出。
- `num_act_out(int)`：输出激活数，与 `get_indice_pairs` 返回的 `out_inds.size(0)` 一致。
- `inverse(int)`：`0` 表示正向卷积，`1` 表示 inverse 卷积（如 sparse deconv）。forward 与 backward 必须传入相同的值；backward 会根据该标志选择输入特征和输出梯度的索引方向。
- `sub_m(int)`：`0` / `1`，需与生成索引对时的 `subm(bool)` 设置一致；这是原生卷积接口的命名，索引生成封装使用 `subm`。
- `out_grad(Tensor)`：反向上游梯度，shape `[numActOut, numOutPlanes]`。

## 返回值

- 前向：输出激活，shape `[numActOut, numOutPlanes]`。
- 反向：`(input_grad, filters_grad)`，shape 分别与 `features`、`filters` 一致。

## 约束说明

- 同时提供 CPU 与 HCU（ROCm/HIP）实现。
- `features` 必须为 2 维；`filters` 通道维需与 `features.size(1)` 对齐。
- `sub_m=1` 时，对有效索引对数量最多的核偏移直接执行矩阵乘；在典型子流形卷积中，该偏移通常对应中心位置。
- fp32 与 half 是两套独立接口，dtype 不匹配不会自动转换。

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
    indices, 1, [5, 5, 5], [5, 5, 5], [3, 3, 3], [1, 1, 1],
    [1, 1, 1], [1, 1, 1], [0, 0, 0], 1, 0,
)
features = torch.randn(
    4,
    16,
    device="cuda",
    dtype=torch.float32,
)
filters = torch.randn(
    3,
    3,
    3,
    16,
    32,
    device="cuda",
    dtype=torch.float32,
)
out = ops.indice_conv_fp32(
    features,
    filters,
    indice_pairs,
    indice_num,
    out_inds.size(0),
    0,
    1,
)
torch.cuda.synchronize()
grad_features, grad_filters = ops.indice_conv_backward_fp32(
    features,
    filters,
    torch.ones_like(out),
    indice_pairs,
    indice_num,
    0,
    1,
)
torch.cuda.synchronize()
print(
    f"features shape: {features.shape}, features dtype: {features.dtype}, "
    f"filters shape: {filters.shape}, filters dtype: {filters.dtype}, "
    f"out shape: {out.shape}, out dtype: {out.dtype}, "
    f"grad_features shape: {grad_features.shape}, "
    f"grad_features dtype: {grad_features.dtype}, "
    f"grad_filters shape: {grad_filters.shape}, "
    f"grad_filters dtype: {grad_filters.dtype}"
)
```

一次运行的输出示例：

```text
features shape: torch.Size([4, 16]), features dtype: torch.float32, filters shape: torch.Size([3, 3, 3, 16, 32]), filters dtype: torch.float32, out shape: torch.Size([4, 32]), out dtype: torch.float32, grad_features shape: torch.Size([4, 16]), grad_features dtype: torch.float32, grad_filters shape: torch.Size([3, 3, 3, 16, 32]), grad_filters dtype: torch.float32
```

## 参考

- 源码：`kernel/sparse_conv/include/spconv/spconv_ops.h`、`kernel/sparse_conv/include/spconv/fused_spconv_ops.h`
- 绑定：`kernel/sparse_conv/src/all.cc`
- 相关接口：[get_indice_pairs](./get_indice_pairs.md)、[indice_maxpool](./indice_maxpool.md)
- 返回[算子 API 清单](../README.md)
