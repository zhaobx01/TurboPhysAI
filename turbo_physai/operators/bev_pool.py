# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""BEV pooling compatibility operators backed by the bundled native kernel."""

from __future__ import annotations

import torch


def _ops():
    from turbo_physai import ops

    return ops


def bev_pool_forward(x, geom_feats, interval_lengths, interval_starts, batch, depth, height, width):
    return _ops().bev_pool_forward(
        x, geom_feats, interval_lengths, interval_starts,
        int(batch), int(depth), int(height), int(width),
    )


def bev_pool_backward(out_grad, geom_feats, interval_lengths, interval_starts, batch, depth, height, width):
    return _ops().bev_pool_backward(
        out_grad, geom_feats, interval_lengths, interval_starts,
        int(batch), int(depth), int(height), int(width),
    )


class BevPoolFunction(torch.autograd.Function):
    """Autograd-compatible wrapper for the native BEV pooling kernel."""

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, batch, depth, height, width):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        if x.shape[0] > 1:
            kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        if interval_starts.numel() > 1:
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        if interval_starts.numel():
            interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        output = bev_pool_forward(
            x, geom_feats.int().contiguous(), interval_lengths, interval_starts,
            batch, depth, height, width,
        )
        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats.int())
        ctx.output_shape = int(batch), int(depth), int(height), int(width)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        starts, lengths, geom_feats = ctx.saved_tensors
        batch, depth, height, width = ctx.output_shape
        grad_x = bev_pool_backward(
            grad_output.contiguous(), geom_feats, lengths, starts,
            batch, depth, height, width,
        )
        return grad_x, None, None, None, None, None, None


def bev_pool(feats, coords, batch, depth, height, width, ranks=None):
    """Pool point features into ``[B, C, D, H, W]`` BEV output."""

    if feats.shape[0] != coords.shape[0]:
        raise ValueError("features and coordinates must have equal rows")
    if ranks is None:
        ranks = (
            coords[:, 0] * (width * depth * batch)
            + coords[:, 1] * (depth * batch)
            + coords[:, 2] * batch
            + coords[:, 3]
        )
    order = ranks.argsort()
    output = BevPoolFunction.apply(
        feats.index_select(0, order),
        coords.index_select(0, order),
        ranks.index_select(0, order),
        batch, depth, height, width,
    )
    return output.permute(0, 4, 1, 2, 3).contiguous()


def bev_pool_prepare(geom_feats, bx, dx, nx, batch, depth, height, width):
    return _ops().bev_pool_prepare(
        geom_feats.contiguous(), bx.contiguous(), dx.contiguous(), nx.contiguous(),
        int(batch), int(depth), int(height), int(width),
    )


def bev_pool_prepare_geometry(
    frustum, inv_post_rots, post_trans, combine, camera2lidar_trans,
    extra_rots, extra_trans, bx, dx, nx, batch, depth, height, width,
    boundary_eps=1.0e-3,
):
    return _ops().bev_pool_prepare_geometry(
        frustum.contiguous(), inv_post_rots.contiguous(), post_trans.contiguous(),
        combine.contiguous(), camera2lidar_trans.contiguous(), extra_rots.contiguous(),
        extra_trans.contiguous(), bx.contiguous(), dx.contiguous(), nx.contiguous(),
        int(batch), int(depth), int(height), int(width), float(boundary_eps),
    )


__all__ = [
    "BevPoolFunction", "bev_pool", "bev_pool_forward", "bev_pool_backward",
    "bev_pool_prepare", "bev_pool_prepare_geometry",
]
