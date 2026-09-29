# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility operator frontends backed by TurboPhysAI native providers.

Operator modules are imported lazily, the same way ``turbo_physai/__init__.py``
exposes them: most of them load Torch at import time, and planning/checking must
stay usable without it.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_OPERATOR_MODULES = {
    "turbo_physai.operators.bev_pool": (
        "BevPoolFunction",
        "bev_pool",
        "bev_pool_forward",
        "bev_pool_backward",
        "bev_pool_prepare",
        "bev_pool_prepare_geometry",
    ),
    "turbo_physai.operators.sparse_conv": (
        "IndiceConvFunction",
        "IndiceMaxPoolFunction",
        "fused_indice_conv_fp32",
        "fused_indice_conv_half",
        "get_indice_pairs",
        "get_indice_pairs_2d",
        "get_indice_pairs_3d",
        "get_indice_pairs_4d",
        "get_indice_pairs_grid_2d",
        "get_indice_pairs_grid_3d",
        "indice_conv_backward_fp32",
        "indice_conv_backward_half",
        "indice_conv_fp32",
        "indice_conv_half",
        "indice_maxpool_backward_fp32",
        "indice_maxpool_backward_half",
        "indice_maxpool_fp32",
        "indice_maxpool_half",
        "indice_conv",
        "indice_maxpool",
    ),
    "turbo_physai.operators.voxelization": (
        "DynamicScatterFunction",
        "dynamic_point_to_voxel_backward",
        "dynamic_point_to_voxel_forward",
        "dynamic_scatter",
        "dynamic_voxelize",
        "hard_voxelize",
        "voxelization_forward",
        "voxelize",
    ),
}

_LAZY_OPERATORS = {
    name: module_name
    for module_name, names in _OPERATOR_MODULES.items()
    for name in names
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_OPERATORS[name]
    except KeyError:
        # Submodules are not in the map. The previous eager ``from .x import``
        # bound them as package attributes, so keep that spelling working too.
        try:
            value = import_module(f"{__name__}.{name}")
        except ImportError as exc:
            raise AttributeError(name) from exc
    else:
        value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "BevPoolFunction",
    "bev_pool",
    "bev_pool_forward",
    "bev_pool_backward",
    "bev_pool_prepare",
    "bev_pool_prepare_geometry",
    "dynamic_voxelize",
    "hard_voxelize",
    "voxelize",
    "voxelization_forward",
    "DynamicScatterFunction",
    "dynamic_point_to_voxel_forward",
    "dynamic_point_to_voxel_backward",
    "dynamic_scatter",
    "get_indice_pairs",
    "get_indice_pairs_2d",
    "get_indice_pairs_3d",
    "get_indice_pairs_4d",
    "get_indice_pairs_grid_2d",
    "get_indice_pairs_grid_3d",
    "IndiceConvFunction",
    "indice_conv",
    "indice_conv_fp32",
    "indice_conv_backward_fp32",
    "indice_conv_half",
    "indice_conv_backward_half",
    "fused_indice_conv_fp32",
    "fused_indice_conv_half",
    "indice_maxpool_fp32",
    "indice_maxpool_backward_fp32",
    "indice_maxpool_half",
    "indice_maxpool_backward_half",
    "IndiceMaxPoolFunction",
    "indice_maxpool",
]
