# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Match-cost replacements for MapTRv2.

``OrderedPtsL1Cost`` calls ``torch.cdist(..., p=1)``.  The ROCm/MIOpen
``cdist`` kernel is only partially covered, its backward path is unstable and it
is opaque to Dynamo.  Replacing it with an explicit broadcast subtraction keeps
the same value while running on the generic elementwise/reduction kernels that
Inductor can also fuse.
"""

import torch


def ordered_pts_l1_cost_call(self, bbox_pred, gt_bboxes):
    """Broadcast L1 cost, equivalent to ``torch.cdist(..., p=1)``."""

    num_gts, num_orders = gt_bboxes.shape[0], gt_bboxes.shape[1]
    bbox_pred = bbox_pred.reshape(bbox_pred.size(0), -1)
    gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
    bbox_cost = (bbox_pred[:, None, :] - gt_bboxes[None, :, :]).abs().sum(-1)
    return bbox_cost * self.weight
