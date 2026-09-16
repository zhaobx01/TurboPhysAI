# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""MMDetection3D-compatible voxelization operators."""

from __future__ import annotations

import torch


def _ops():
    from turbo_physai import ops

    return ops


def dynamic_voxelize(points, voxel_size, coors_range, ndim=3):
    """Return one integer voxel coordinate for each input point."""

    coords = points.new_zeros((points.size(0), int(ndim)), dtype=torch.int)
    _ops().dynamic_voxelize(
        points, coords, voxel_size, coors_range, int(ndim)
    )
    return coords


def hard_voxelize(
    points, voxel_size, coors_range, max_points=35, max_voxels=20000,
    ndim=3, deterministic=True,
):
    """Group points into bounded voxels using the bundled native kernel."""

    voxels = points.new_zeros((max_voxels, max_points, points.size(1)))
    coords = points.new_zeros((max_voxels, int(ndim)), dtype=torch.int)
    counts = points.new_zeros((max_voxels,), dtype=torch.int)
    count = _ops().hard_voxelize(
        points, voxels, coords, counts, voxel_size, coors_range,
        int(max_points), int(max_voxels), int(ndim), bool(deterministic),
    )
    return voxels[:count], coords[:count], counts[:count]


def dynamic_point_to_voxel_forward(feats, coors, reduce_type="max"):
    """Expose the native DynamicScatter forward contract."""

    if reduce_type not in {"max", "sum", "mean"}:
        raise ValueError(f"unsupported reduce type: {reduce_type}")
    return _ops().dynamic_point_to_voxel_forward(
        feats.contiguous(), coors.int().contiguous(), reduce_type
    )


def dynamic_point_to_voxel_backward(
    grad_feats,
    grad_reduced_feats,
    feats,
    reduced_feats,
    coors_idx,
    reduce_count,
    reduce_type="max",
):
    """Write DynamicScatter input gradients into ``grad_feats``."""

    if reduce_type not in {"max", "sum", "mean"}:
        raise ValueError(f"unsupported reduce type: {reduce_type}")
    _ops().dynamic_point_to_voxel_backward(
        grad_feats,
        grad_reduced_feats.contiguous(),
        feats,
        reduced_feats,
        coors_idx,
        reduce_count,
        reduce_type,
    )


class DynamicScatterFunction(torch.autograd.Function):
    """MMDetection3D-style differentiable point-to-voxel reduction."""

    @staticmethod
    def forward(ctx, feats, coors, reduce_type="max"):
        reduced, out_coors, coors_idx, reduce_count = (
            dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        )
        ctx.save_for_backward(feats, reduced, coors_idx, reduce_count)
        ctx.reduce_type = reduce_type
        ctx.mark_non_differentiable(out_coors)
        return reduced, out_coors

    @staticmethod
    def backward(ctx, grad_reduced, grad_coors=None):
        del grad_coors
        feats, reduced, coors_idx, reduce_count = ctx.saved_tensors
        grad_feats = torch.empty_like(feats)
        dynamic_point_to_voxel_backward(
            grad_feats,
            grad_reduced,
            feats,
            reduced,
            coors_idx,
            reduce_count,
            ctx.reduce_type,
        )
        return grad_feats, None, None


def dynamic_scatter(feats, coors, reduce_type="max"):
    """Reduce point features by coordinate, returning features and coordinates."""

    return DynamicScatterFunction.apply(feats, coors, reduce_type)


def voxelize(
    points, voxel_size, coors_range, max_points=35, max_voxels=20000,
    deterministic=True,
):
    """Compatibility frontend matching MMDetection3D voxelization output."""

    if max_points == -1 or max_voxels == -1:
        return dynamic_voxelize(points, voxel_size, coors_range, 3)
    return hard_voxelize(
        points, voxel_size, coors_range, max_points, max_voxels, 3, deterministic
    )


def voxelization_forward(
    ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000,
    deterministic=True,
):
    """Match ``_Voxelization.forward(ctx, ...)`` without retaining ``ctx``."""

    del ctx
    return voxelize(
        points, voxel_size, coors_range, max_points, max_voxels, deterministic
    )


__all__ = [
    "DynamicScatterFunction",
    "dynamic_point_to_voxel_forward",
    "dynamic_point_to_voxel_backward",
    "dynamic_scatter",
    "dynamic_voxelize",
    "hard_voxelize",
    "voxelize",
    "voxelization_forward",
]
