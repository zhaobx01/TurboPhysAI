# dynamic_voxelize

## 接口原型

```python
turbo_physai.ops.dynamic_voxelize(
    points, coors, voxel_size, coors_range, NDim
) -> None
```

接口位置：`turbo_physai.ops`。上层封装见 `turbo_physai/optimizations/common/mmdet3d/voxelization.py`。

## 功能描述

动态体素化：为每个点单独计算所属体素坐标，结果原地写入 `coors`，不做体素内聚合、也不统计点数。用于需要"点级"体素索引的场景，例如 DynamicScatter 之前的坐标映射。

## 参数说明

- `points(Tensor)`：float32，shape `[N, F]`。
- `coors(Tensor)`：**输出缓冲区**，需预先分配，int32，shape `[N, NDim]`。
- `voxel_size(List[float])`：长度 `NDim`。
- `coors_range(List[float])`：长度 `2 * NDim`，前半为各维最小值、后半为最大值。
- `NDim(int)`：坐标维数，通常 3。

## 返回值

无返回值，结果原地写入 `coors`。落入范围的点写入对应体素坐标；越界点至少会将 `coors[:, 0]` 置为 `-1`，判断有效点时应检查首列。不同 CPU/GPU 实现对其余列的写入可能不同。

## 约束说明

- 同时提供 CPU 与 HCU（ROCm/HIP）实现；GPU 路径要求输入连续。
- 内核不会自动扩容 `coors`，需按 `[N, NDim]` 预先分配。

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

coors = points.new_zeros(
    (points.size(0), 3),
    dtype=torch.int32,
)
voxel_size = [0.05, 0.05, 0.1]
coors_range = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]

# 原地计算每个点对应的体素坐标
ops.dynamic_voxelize(
    points,
    coors,
    voxel_size,
    coors_range,
    3,
)

# 过滤越界点
torch.cuda.synchronize()
valid = coors[:, 0] >= 0

print(
    f"points shape: {points.shape}, points dtype: {points.dtype}, "
    f"coors shape: {coors.shape}, coors dtype: {coors.dtype}, "
    f"valid points: {valid.sum().item()}, "
    f"invalid points: {(~valid).sum().item()}"
)
```

一次运行的输出示例：

```text
points shape: torch.Size([20000, 3]), points dtype: torch.float32, coors shape: torch.Size([20000, 3]), coors dtype: torch.int32, valid points: 151, invalid points: 19849
```

## 参考

- 源码：`kernel/voxelization/src/voxelization.h`、`kernel/voxelization/src/voxelization_cpu.cpp`、`kernel/voxelization/src/voxelization_cuda.cu`
- 相关接口：[hard_voxelize](./hard_voxelize.md)
- 返回[算子 API 清单](../README.md)