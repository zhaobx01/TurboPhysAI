# 算子 API 清单

本目录记录 `kernel/` 中原生算子的 Python 接口，供算子调试、精度对比和二次开发时查阅。

> 接口原型已与源码核对，如有不一致，以源码为准。

日常使用 TurboPhysAI 时不需要手工调用这些接口。通过 `turbo_physai.apply()` 加载配置后，上游算子会自动替换为这里的实现，参考[优化配置](../user_guide/optimization_config.md)。

## 调用方式

以下调用示例均针对已编译 GPU 扩展的 HCU/ROCm 环境；在 PyTorch ROCm 环境中通常仍使用 `device="cuda"` 表示 GPU。

原生算子编译在 `turbo_physai.ops` 扩展模块中，构建方式见[安装指导](../get_started/installation.md)：

```python
from turbo_physai import ops

out = ops.bev_pool_forward(x, geom_feats, interval_lengths, interval_starts, B, D, H, W)
```

本目录的接口示例统一直接调用 `turbo_physai.ops` 原生接口。

## API 清单

| 分类 | API | 接口位置 | 说明 |
|---|---|---|---|
| 特征融合 | [bev_pool_forward / bev_pool_backward](./context/bev_pool.md) | `turbo_physai.ops` | BEV 池化前向与反向 |
| 特征融合 | [bev_pool_prepare / bev_pool_prepare_geometry](./context/bev_pool.md) | `turbo_physai.ops` | BEV 池化坐标与排序准备 |
| 采样 | [deformable_aggregation_forward / backward](./context/deformable_aggregation.md) | `turbo_physai.ops` | 可变形聚合 |
| 采样 | [grid_sample_forward / backward](./context/grid_sample.md) | `turbo_physai.ops` | 2D 双线性网格采样 |
| 采样 | [upsample_bilinear_2d_forward / backward](./context/upsample_bilinear_2d.md) | `turbo_physai.ops` | 2D 双线性上采样 |
| 体素化 | [hard_voxelize](./context/hard_voxelize.md) | `turbo_physai.ops` | 固定体素数的硬体素化 |
| 体素化 | [dynamic_voxelize](./context/dynamic_voxelize.md) | `turbo_physai.ops` | 动态体素化 |
| 体素化 | [dynamic_point_to_voxel_forward / backward](./context/dynamic_point_to_voxel.md) | `turbo_physai.ops` | 点到体素归约与其反向 |
| 稀疏卷积 | [get_indice_pairs](./context/get_indice_pairs.md) | `turbo_physai.ops` | 稀疏卷积索引对生成 |
| 稀疏卷积 | [indice_conv / fused_indice_conv](./context/indice_conv.md) | `turbo_physai.ops` | 稀疏卷积与融合版本 |
| 稀疏卷积 | [indice_maxpool](./context/indice_maxpool.md) | `turbo_physai.ops` | 稀疏最大池化与其反向 |

## 与 kernel/ 目录的对应关系

| kernel 目录 | 对应 API |
|---|---|
| `kernel/bev_pool/` | `bev_pool_forward`、`bev_pool_backward`、`bev_pool_prepare`、`bev_pool_prepare_geometry` |
| `kernel/deformable_aggregation/` | `deformable_aggregation_forward`、`deformable_aggregation_backward` |
| `kernel/grid_sample/` | `grid_sample_forward`、`grid_sample_backward` |
| `kernel/upsample_bilinear_2d/` | `upsample_bilinear_2d_forward`、`upsample_bilinear_2d_backward` |
| `kernel/voxelization/` | `hard_voxelize`、`dynamic_voxelize`、`dynamic_point_to_voxel_forward`、`dynamic_point_to_voxel_backward` |
| `kernel/sparse_conv/` | `get_indice_pairs_*`、`indice_conv_*`、`fused_indice_conv_*`、`indice_maxpool_*` |

`kernel/common/` 只提供公共头文件，不对外暴露算子。

## 相关算子

以下算子面向模型提供，但不位于 `kernel/`：

- `modulated_deform_conv2d`、`MultiScaleDeformableAttnFunction`：实现在 `turbo_physai/operators/`，计算由 hipdnn 提供。

## 约束汇总

- 以下示例默认使用已编译的 HCU/ROCm GPU 扩展；部分算子同时提供 CPU 路径，具体设备支持以各算子文档为准。
- 多数算子输入需要 `float32`；稀疏卷积的 `indice_conv_half`、`indice_maxpool_half` 额外支持 `float16`。
- 传入张量需满足各接口文档中列出的 shape、dtype 与连续性要求。
