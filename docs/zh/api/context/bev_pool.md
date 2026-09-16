# bev_pool

## 接口原型

```python
turbo_physai.ops.bev_pool_forward(
    x, geom_feats, interval_lengths, interval_starts, b, d, h, w
) -> Tensor
turbo_physai.ops.bev_pool_backward(
    out_grad, geom_feats, interval_lengths, interval_starts, b, d, h, w
) -> Tensor
turbo_physai.ops.bev_pool_prepare(
    geom_feats, bx, dx, nx, b, d, h, w
) -> Tuple[Tensor, Tensor, Tensor]
turbo_physai.ops.bev_pool_prepare_geometry(
    frustum, inv_post_rots, post_trans, combine, camera2lidar_trans,
    extra_rots, extra_trans, bx, dx, nx, b, d, h, w, boundary_eps
) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

接口位置：`turbo_physai.ops`。以下示例直接调用 native extension；可选的模型兼容辅助层见 `turbo_physai/operators/bev_pool.py`。

## 功能描述

- `bev_pool_forward` / `bev_pool_backward`：BEV 特征池化的前向与反向。把同一 BEV 网格内多个深度采样点的特征累加到一个体素。
- `bev_pool_prepare`：把浮点几何坐标一次性转换为整数体素坐标、排序秩和有效掩码。
- `bev_pool_prepare_geometry`：独立的几何准备接口，将视锥体和相机变换转换为 BEV 网格坐标，并额外输出边界候选。

## 参数说明

`bev_pool_forward` / `bev_pool_backward`：

- `x(Tensor)`：float32，shape `[N, C]`，需已按 `ranks` 升序排好。
- `geom_feats(Tensor)`：int32，shape `[N, 4]`，每行为 `(height, width, depth, batch)`；对应输出布局 `[B, D, H, W, C]`。
- `interval_lengths, interval_starts(Tensor)`：int32，shape `[n_intervals]`，由调用方根据 `ranks` 计算。
- `b, d, h, w(int)`：输出 BEV 网格大小。
- `out_grad(Tensor)`：反向输入，float32，shape `[b, d, h, w, C]`。

`bev_pool_prepare`：

- `geom_feats(Tensor)`：float32，shape `[B, N_cam, D_geom, H_geom, W_geom, 3]`，其中 `N_cam` 为相机/视图维度，最后一维为 `(x, y, z)`。
- `bx / dx(Tensor)`：float32，长度 3，分别为体素原点与尺寸。
- `nx(Tensor)`：int64，长度 3，各方向体素数。
- `b, d, h, w(int)`：输出 BEV 网格大小。

`bev_pool_prepare_geometry` 还需要以下相机几何参数：

- `frustum(Tensor)`：float32，shape `[D, H, W, 3]`。
- `inv_post_rots(Tensor)`：float32，shape `[B, N, 3, 3]`。
- `post_trans, combine, camera2lidar_trans, extra_rots, extra_trans(Tensor)`：其余几何参数，float32。
- `boundary_eps(float)`：边界判定容差，调用示例使用 `1e-3`。

## 返回值

- `bev_pool_forward`：float32，shape `[B, D, H, W, C]`。
- `bev_pool_backward`：float32，shape `[N, C]`。
- `bev_pool_prepare`：`(coords, ranks, kept)`，分别为 int32 `[total, 4]`、int32 `[total]`、bool `[total]`。
- `bev_pool_prepare_geometry`：`(coords, ranks, kept, boundary)`，前三者同上，`boundary` 为 bool `[total]`。

其中 `total = geom_feats.numel() / 3`，即所有采样点总数。

## 约束说明

- 所有 BEV pool native 接口均面向 CUDA/HIP 张量；`bev_pool_prepare` 与 `bev_pool_prepare_geometry` 另外要求 `geom_feats` 与 `frustum` 为 float32，shape 不符合将直接报错。
- `bev_pool_forward` 假定 `x` 已按 `ranks` 排序，`interval_starts / interval_lengths` 与排序结果一致，由调用方自行保证。
- 输出通道数取自 `x.size(1)`，`geom_feats` 只用于定位。

## 调用示例

排序与池化，参考 `turbo_physai/operators/bev_pool.py`。当前 Kernel 使用的
`geom_feats` 布局为 `(height, width, depth, batch)`；输入必须先按照同一组
`ranks` 排序，使相同 BEV 位置的数据连续排列：

```python
import torch
from turbo_physai import ops
B, D, H, W, C = 1, 4, 128, 128, 64
N = 1024
x = torch.randn(N, C, device="cuda", dtype=torch.float32)
geom_feats = torch.empty(N, 4, device="cuda", dtype=torch.int32)
geom_feats[:, 0] = torch.randint(0, H, (N,), device="cuda")
geom_feats[:, 1] = torch.randint(0, W, (N,), device="cuda")
geom_feats[:, 2] = torch.randint(0, D, (N,), device="cuda")
geom_feats[:, 3] = torch.randint(0, B, (N,), device="cuda")
ranks = (
    ((geom_feats[:, 3].to(torch.int64) * D
      + geom_feats[:, 2].to(torch.int64)) * H
     + geom_feats[:, 0].to(torch.int64)) * W
    + geom_feats[:, 1].to(torch.int64)
)
order = torch.argsort(ranks)
x = x[order].contiguous()
geom_feats = geom_feats[order].contiguous()
ranks = ranks[order].contiguous()
kept = torch.ones(N, device="cuda", dtype=torch.bool)
kept[1:] = ranks[1:] != ranks[:-1]
interval_starts = torch.where(kept)[0].to(torch.int32)
interval_lengths = torch.empty_like(interval_starts)
if interval_starts.numel() > 1:
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
interval_lengths[-1] = N - interval_starts[-1]
out = ops.bev_pool_forward(
    x,
    geom_feats,
    interval_lengths,
    interval_starts,
    B,
    D,
    H,
    W,
)
torch.cuda.synchronize()
out_grad = torch.ones_like(out)
x_grad = ops.bev_pool_backward(
    out_grad,
    geom_feats,
    interval_lengths,
    interval_starts,
    B,
    D,
    H,
    W,
)
torch.cuda.synchronize()
print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"x_grad shape: {x_grad.shape}, x_grad dtype: {x_grad.dtype}"
)
```
一次运行的输出示例：
```text
output shape: torch.Size([1, 4, 128, 128, 64]), output dtype: torch.float32, x_grad shape: torch.Size([1024, 64]), x_grad dtype: torch.float32
```

## 参考

- 源码：`kernel/bev_pool/src/bev_pool_cpu.cpp`、`kernel/bev_pool/src/bev_pool_cuda.cu`
- 可选的模型兼容辅助层：`turbo_physai/operators/bev_pool.py`
- 返回[算子 API 清单](../README.md)
