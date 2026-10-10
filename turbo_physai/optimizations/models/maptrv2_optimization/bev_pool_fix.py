# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""ROCm bev_pool layout fix for MapTRv2.

ROCm/HIP bev_pool returns [B, Z, H, W, C] (channels-last) instead of
[B, C, Z, H, W] as the CUDA version does.  The collapse-Z cat that follows
therefore receives the wrong axis and produces a misshapen tensor, causing the
downstream Conv2d to raise a channel-count mismatch.

The replacement returns the baseline's collapsed 4D layout by default. The
reference-boundaries path requests the uncollapsed 5D layout through
``return_5d=True`` so both Groups use the same pooling implementation.
"""

import torch
from mmdet3d.ops import bev_pool as _bev_pool


def base_transform_bev_pool(self, geom_feats, x, *, return_5d=False):
    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W

    # flatten x
    x = x.reshape(Nprime, C)

    # flatten indices
    geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
    geom_feats = geom_feats.view(Nprime, 3)
    batch_ix = torch.cat(
        [
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ]
    )
    geom_feats = torch.cat((geom_feats, batch_ix), 1)

    # filter out points that are outside box
    kept = (
        (geom_feats[:, 0] >= 0)
        & (geom_feats[:, 0] < self.nx[0])
        & (geom_feats[:, 1] >= 0)
        & (geom_feats[:, 1] < self.nx[1])
        & (geom_feats[:, 2] >= 0)
        & (geom_feats[:, 2] < self.nx[2])
    )
    x = x[kept]
    geom_feats = geom_feats[kept]

    x = _bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

    # The reference boundary keeps Z separate until its compiled down_sample
    # helper. The standalone replacement retains the baseline's 4D contract.
    if return_5d:
        expected_nhwc = (B, int(self.nx[2]), int(self.nx[0]), int(self.nx[1]), C)
        if tuple(x.shape) == expected_nhwc:
            return x
        expected_nchw = (B, C, int(self.nx[2]), int(self.nx[0]), int(self.nx[1]))
        if tuple(x.shape) == expected_nchw:
            return x.permute(0, 2, 3, 4, 1).contiguous()
        raise RuntimeError(
            f"cannot identify BEV channel dimension in shape {tuple(x.shape)}"
        )

    # ROCm/HIP bev_pool may return [B, Z, H, W, C] instead of [B, C, Z, H, W]
    if x.dim() == 5 and x.shape[-1] == self.C:
        x = x.permute(0, 4, 1, 2, 3).contiguous()

    # collapse Z
    final = torch.cat(x.unbind(dim=2), 1)
    return final


# BaseTransform.forward can ask this replacement for the 5D layout when the
# reference boundaries Group is active. Other bev_pool implementations fall
# back to reference._bev_pool_5d.
base_transform_bev_pool._turbo_physai_returns_5d = True
