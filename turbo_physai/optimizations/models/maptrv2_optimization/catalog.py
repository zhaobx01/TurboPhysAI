# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""MapTRv2 optimization declarations.

Every Group below maps one item of the validated MapTRv2 reference optimization
onto a symbol that exists in the official ``maptrv2`` branch, so the recipe can
be applied to a clean baseline checkout:

===========================================  ==================================
Group                                        Reference change
===========================================  ==================================
``maptrv2.training``                         channels-last backbone, cuDNN
                                             benchmark/非确定性, ``fork`` start
                                             method
``maptrv2.data``                             ``pin_memory=True``
``maptrv2.grid_mask``                        Dynamo-safe ``GridMask.forward``
``maptrv2.compile``                          ``torch.compile`` on every baseline
                                             symbol carrying a compile hook in
                                             the reference
``maptrv2.match_cost``                       ``torch.cdist(p=1)`` replaced by a
                                             broadcast subtraction
``maptrv2.pv_mask``                          ``line_ego_to_pvmask`` drops the
                                             shapely interpolation loop and
                                             ``line_ego_to_mask`` merges the
                                             two affine transform steps;
                                             ``gen_vectorized_samples`` drops one
                                             vertex copy
``maptrv2.assigner``                         padded GT contract and Dynamo
                                             isolation for Hungarian matching
``maptrv2.efficientnet``                     ``EfficientNet`` allowed to
                                             override the registry entry
``maptrv2.spconv_registry``                  SparseConv classes allowed to
                                             override registry entries
``maptrv2.reference_boundaries``             eight extracted helper boundaries
                                             plus the reference BEV layout
===========================================  ==================================

``maptrv2.training`` also pins the fp32 matmul precision to the reference's
``"high"``.

``optimization_modules`` loads this module, so importing it registers every
Group with TurboPhysAI.
"""

from __future__ import annotations

from turbo_physai import group, replace, wrap
from turbo_physai.compatibility import registry_override


_TRAIN_API = (
    "projects.mmdet3d_plugin.bevformer.apis.mmdet_train.custom_train_detector"
)
_BUILD_DATALOADER_API = "projects.mmdet3d_plugin.datasets.builder.build_dataloader"
_GRID_MASK_API = "projects.mmdet3d_plugin.models.utils.grid_mask.GridMask.forward"
_ASSIGN_API = (
    "projects.mmdet3d_plugin.maptr.assigners.maptr_assigner.MapTRAssigner.assign"
)

# ``torch.compile``/``torch._dynamo`` need a new enough Torch and the user's
# consent, so the Groups that depend on them dispatch per call instead of
# failing while ``apply`` installs them.  See ``turbo_physai.optimizations.models.maptrv2_optimization.compat``.
_COMPILE_CONDITION = "turbo_physai.optimizations.models.maptrv2_optimization.compat.torch_compile_available"
_STATIC_ASSIGNER_CONDITION = (
    "turbo_physai.optimizations.models.maptrv2_optimization.compat."
    "static_assigner_available"
)
_PV_MASK_CONDITION = "turbo_physai.optimizations.models.maptrv2_optimization.compat.pv_mask_sampling_enabled"


TRAINING = group(
    "maptrv2.training",
    wrap(
        target=_TRAIN_API,
        aliases=("projects.mmdet3d_plugin.bevformer.apis.custom_train_detector",),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization.training.training_runtime_wrapper"
        ),
    ),
)

DATA = group(
    "maptrv2.data",
    replace(
        target=_BUILD_DATALOADER_API,
        aliases=(
            "projects.mmdet3d_plugin.bevformer.apis.mmdet_train.build_dataloader",
        ),
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.data.build_dataloader",
    ),
)

GRID_MASK = group(
    "maptrv2.grid_mask",
    replace(
        target=_GRID_MASK_API,
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.grid_mask.grid_mask_forward",
    ),
)

# The reference carries ten compile hooks on baseline symbols listed below.
# Its eight additional helper hooks are implemented by the
# ``maptrv2.reference_boundaries`` Group, which replaces the existing callers.
_COMPILE_TARGETS = (
    "projects.mmdet3d_plugin.maptr.modules.transformer."
    "MapTRPerceptionTransformer.format_feats",
    "projects.mmdet3d_plugin.maptr.modules.decoder.MapTRDecoder.forward",
    "projects.mmdet3d_plugin.maptr.detectors.maptrv2.MapTRv2.extract_img_feat",
    "projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.get_cam_feats",
    "projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.get_mlp_input",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_3d_pts",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_2d_bbox",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_2d_pts",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
    "denormalize_2d_bbox",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
    "denormalize_2d_pts",
)

COMPILE = group(
    "maptrv2.compile",
    *(
        wrap(
            target=target,
            replacement="turbo_physai.optimizations.models.maptrv2_optimization.compile.compile_wrapper",
            runtime_condition=_COMPILE_CONDITION,
        )
        for target in _COMPILE_TARGETS
    ),
)

MATCH_COST = group(
    "maptrv2.match_cost",
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.losses.map_loss."
            "OrderedPtsL1Cost.__call__"
        ),
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.match_cost.ordered_pts_l1_cost_call",
    ),
)

PV_MASK = group(
    "maptrv2.pv_mask",
    replace(
        target=(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_pvmask"
        ),
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.pv_mask.line_ego_to_pvmask",
        runtime_condition=_PV_MASK_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_mask"
        ),
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.pv_mask.line_ego_to_mask",
        runtime_condition=_PV_MASK_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.gen_vectorized_samples"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "pv_mask.gen_vectorized_samples"
        ),
        runtime_condition=_PV_MASK_CONDITION,
    ),
)

ASSIGNER = group(
    "maptrv2.assigner",
    replace(
        target=_ASSIGN_API,
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "assigner.assign"
        ),
        runtime_condition=_STATIC_ASSIGNER_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head.loss"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "assigner.static_loss"
        ),
        runtime_condition=_STATIC_ASSIGNER_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head._get_target_single"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "assigner.get_target_single"
        ),
        runtime_condition=_STATIC_ASSIGNER_CONDITION,
    ),
)

EFFICIENTNET = group(
    "maptrv2.efficientnet",
    registry_override(
        module="projects.mmdet3d_plugin.models.backbones.efficientnet",
        registry="mmdet.models.builder.BACKBONES",
        names=("EfficientNet",),
    ),
)

BEV_POOL_FIX = group(
    "maptrv2.bev_pool_fix",
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder.BaseTransform.bev_pool"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization"
            ".bev_pool_fix.base_transform_bev_pool"
        ),
    ),
)

REFERENCE_BOUNDARIES = group(
    "maptrv2.reference_boundaries",
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "BaseTransform.get_geometry_v1"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "reference.base_transform_get_geometry_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "BaseTransform.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "reference.base_transform_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "LSSTransform.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "reference.ls_transform_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.transformer."
            "MapTRPerceptionTransformer.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "reference.transformer_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2_optimization."
            "reference.head_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
)

SPARSE_CONV_REGISTRY = group(
    "maptrv2.spconv_registry",
    registry_override(
        module="mmdet3d.ops.spconv.conv",
        registry="mmcv.cnn.CONV_LAYERS",
        names=(
            "SparseConv2d",
            "SparseConv3d",
            "SparseConv4d",
            "SparseConvTranspose2d",
            "SparseConvTranspose3d",
            "SparseInverseConv2d",
            "SparseInverseConv3d",
            "SubMConv2d",
            "SubMConv3d",
            "SubMConv4d",
        ),
    ),
)

__all__ = [
    "ASSIGNER",
    "BEV_POOL_FIX",
    "COMPILE",
    "DATA",
    "EFFICIENTNET",
    "GRID_MASK",
    "MATCH_COST",
    "PV_MASK",
    "REFERENCE_BOUNDARIES",
    "SPARSE_CONV_REGISTRY",
    "TRAINING",
]
