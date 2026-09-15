# indice_maxpool

## 接口原型

```python
turbo_physai.ops.indice_maxpool_fp32(
    features, indice_pairs, indice_num, num_act
) -> Tensor
turbo_physai.ops.indice_maxpool_backward_fp32(
    features, out_features, out_grad, indice_pairs, indice_num
) -> Tensor

# float16 版本：indice_maxpool_half / indice_maxpool_backward_half
```

接口位置：`turbo_physai.ops`，绑定见 `kernel/sparse_conv/src/all.cc`。

## 功能描述

稀疏最大池化：按索引对在稀疏激活之间取最大值，用于稀疏卷积网络的下采样。反向把梯度回传给前向取到最大值的输入位置。

## 参数说明

前向：

- `features(Tensor)`：输入激活，shape `[numActIn, numInPlanes]`。
- `indice_pairs, indice_num(Tensor)`：`get_indice_pairs` 的输出。
- `num_act(int)`：输出激活数，即输出行数。

反向：

- `features, out_features(Tensor)`：前向输入与输出，用于判定最大值位置。
- `out_grad(Tensor)`：输出侧上游梯度，shape `[num_act, numInPlanes]`。
- `indice_pairs, indice_num(Tensor)`：与前向一致。

## 返回值

- 前向：池化结果，shape `[num_act, numInPlanes]`。
- 反向：输入梯度，shape 与 `features` 一致。

## 约束说明

- 同时提供 CPU 与 HCU（ROCm/HIP）实现。
- 输出以 `0` 初始化后再取最大值，因此语义等价于"与 0 取最大"：全负特征的体素会被归为 `0`。输入可能为负时需自行评估该语义。
- 反向按 `out_features == features` 判定最大值位置，若同一最大值出现在多个输入位置，梯度会累加到所有相等位置。
- `num_act` 需与索引对生成时的输出激活数一致。

## 调用示例

```python
import torch
from turbo_physai import ops
from turbo_physai.optimizations.common.mmdet3d.sparse_conv import (
    get_indice_pairs,
)
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
features = torch.rand(
    4,
    16,
    device="cuda",
    dtype=torch.float32,
)
out_inds, indice_pairs, indice_num = get_indice_pairs(
    indices,
    batch_size=1,
    spatial_shape=[5, 5, 5],
    ksize=3,
    subm=True,
)
out = ops.indice_maxpool_fp32(
    features,
    indice_pairs,
    indice_num,
    out_inds.size(0),
)
torch.cuda.synchronize()
grad_features = ops.indice_maxpool_backward_fp32(
    features,
    out,
    torch.ones_like(out),
    indice_pairs,
    indice_num,
)
torch.cuda.synchronize()
print(
    f"features shape: {features.shape}, features dtype: {features.dtype}, "
    f"out shape: {out.shape}, out dtype: {out.dtype}, "
    f"grad_features shape: {grad_features.shape}, "
    f"grad_features dtype: {grad_features.dtype}"
)
```

一次运行的输出示例：

```text
features shape: torch.Size([4, 16]), features dtype: torch.float32, out shape: torch.Size([4, 16]), out dtype: torch.float32, grad_features shape: torch.Size([4, 16]), grad_features dtype: torch.float32
```

## 参考

- 源码：`kernel/sparse_conv/include/spconv/pool_ops.h`、`kernel/sparse_conv/src/maxpool_cpu.cc`、`kernel/sparse_conv/src/maxpool_cuda.cu`
- 绑定：`kernel/sparse_conv/src/all.cc`
- 相关接口：[get_indice_pairs](./get_indice_pairs.md)、[indice_conv](./indice_conv.md)
- 返回[算子 API 清单](../README.md)
