# hard_voxelize

## 接口原型

```python
turbo_physai.ops.hard_voxelize(
    points, voxels, coors, num_points_per_voxel,
    voxel_size, coors_range, max_points, max_voxels,
    NDim, deterministic
) -> int
```

接口位置：`turbo_physai.ops`。以下示例直接调用 native extension；可选的模型兼容辅助层见 `turbo_physai/operators/voxelization.py`。

## 功能描述

硬体素化：把点云按固定网格划分为体素，每个体素最多保留 `max_points` 个点、总共最多输出 `max_voxels` 个体素。用于 PointPillars、CenterPoint 等模型的点云预处理。

与 `dynamic_voxelize` 的区别：本接口会把同一体素的点连续存放并统计点数；`dynamic_voxelize` 只计算每个点所在的体素坐标。

## 参数说明

- `points(Tensor)`：浮点 Tensor，shape `[N, F]`，`F` 通常为 3；CPU/GPU 路径的具体类型支持以编译后的类型分派为准。
- `voxels, coors, num_points_per_voxel(Tensor)`：三个输出缓冲区，由调用方按下列 shape 预分配（原地写入）：
    - `voxels`：`[max_voxels, max_points, F]`，dtype 与 `points` 一致；
    - `coors`：`[max_voxels, 3]`，int32；
    - `num_points_per_voxel`：`[max_voxels]`，int32。
- `voxel_size(List[float])`：长度 3，对应 `x, y, z`。
- `coors_range(List[float])`：长度 6，依次为 `x_min, y_min, z_min, x_max, y_max, z_max`。
- `max_points, max_voxels(int)`：每体素最大点数、最大体素数。
- `NDim(int)`：坐标维数，通常 3。
- `deterministic(bool)`：`False` 走非确定性快速实现，同一体素内点顺序不保证稳定。

## 返回值

有效体素数 `voxel_count`，输出张量的前 `voxel_count` 行才有效，需自行切片。

## 约束说明

- 同时提供 CPU 与 HCU（ROCm/HIP）实现；GPU 路径要求输入张量连续，CPU/GPU 支持的具体 dtype 以编译后的 dispatch 为准。
- 超出 `max_voxels` 的多余体素会被丢弃；越界点不会被写入。
- HCU 路径默认走排序优化实现，触发条件：`points` 为 float32 二维张量、`max_points > 0`、`max_voxels > 0`、`NDim == 3`。可用 `MMDET3D_DISABLE_VOXELIZE_OPT=1` 关闭，或用 `MMDET3D_VOXELIZE_OPT_MODE` 切换（`off` / `legacy` / `safe` 等）。

## 调用示例

```python
import torch
from turbo_physai import ops
points = torch.rand(
    20000,
    3,
    device="cuda",
    dtype=torch.float32,
) * 70.0
voxel_size = [0.05, 0.05, 0.1]
coors_range = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
max_points, max_voxels = 10, 40000
voxels = points.new_zeros(
    (max_voxels, max_points, points.size(1)),
)
coors = points.new_zeros(
    (max_voxels, 3),
    dtype=torch.int32,
)
point_counts = points.new_zeros(
    (max_voxels,),
    dtype=torch.int32,
)
n = ops.hard_voxelize(
    points,
    voxels,
    coors,
    point_counts,
    voxel_size,
    coors_range,
    max_points,
    max_voxels,
    3,
    True,
)
voxels = voxels[:n]
coors = coors[:n]
point_counts = point_counts[:n]
torch.cuda.synchronize()
print(
    f"points shape: {points.shape}, points dtype: {points.dtype}, "
    f"voxels shape: {voxels.shape}, voxels dtype: {voxels.dtype}, "
    f"coors shape: {coors.shape}, coors dtype: {coors.dtype}, "
    f"point_counts shape: {point_counts.shape}, "
    f"point_counts dtype: {point_counts.dtype}, "
    f"num_voxels: {n}"
)
```

一次运行的输出示例：

```text
points shape: torch.Size([20000, 3]), points dtype: torch.float32, voxels shape: torch.Size([164, 10, 3]), voxels dtype: torch.float32, coors shape: torch.Size([164, 3]), coors dtype: torch.int32, point_counts shape: torch.Size([164]), point_counts dtype: torch.int32, num_voxels: 164
```

## 参考

- 源码：`kernel/voxelization/src/voxelization.h`、`kernel/voxelization/src/voxelization_cuda.cu`、`kernel/voxelization/src/voxelization_cpu.cpp`
- 可选的模型兼容辅助层：`turbo_physai/operators/voxelization.py`
- 相关接口：[dynamic_voxelize](./dynamic_voxelize.md)、[dynamic_point_to_voxel](./dynamic_point_to_voxel.md)
- 返回[算子 API 清单](../README.md)
