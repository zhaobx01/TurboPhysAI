# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Structural tests for the MapTRv2 optimization catalog."""

import importlib
import inspect
import os
import pathlib
import subprocess
import sys
import unittest

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.definitions.registry import default_registry

import turbo_physai.optimizations.models.maptrv2_optimization.catalog as catalog


EXPECTED_GROUPS = {
    "maptrv2.training": Mechanism.WRAPPER,
    "maptrv2.data": Mechanism.REPLACE,
    "maptrv2.grid_mask": Mechanism.REPLACE,
    "maptrv2.compile": Mechanism.WRAPPER,
    "maptrv2.match_cost": Mechanism.REPLACE,
    "maptrv2.pv_mask": Mechanism.REPLACE,
    "maptrv2.assigner": Mechanism.WRAPPER,
    "maptrv2.efficientnet": Mechanism.REGISTRY_OVERRIDE,
}

# Groups whose members must re-check an optional dependency on every call.
# The condition is resolved by the engine and must stay a plain callable.
EXPECTED_CONDITIONS = {
    "maptrv2.compile": (
        "turbo_physai.optimizations.models.maptrv2_optimization.compat.torch_compile_available", 10),
    "maptrv2.assigner": ("turbo_physai.optimizations.models.maptrv2_optimization.compat.dynamo_available", 1),
    "maptrv2.pv_mask": (
        "turbo_physai.optimizations.models.maptrv2_optimization.compat.pv_mask_sampling_enabled", 2),
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class CatalogTest(unittest.TestCase):
    def test_catalog_can_be_imported(self):
        importlib.import_module("turbo_physai.optimizations.models.maptrv2_optimization.catalog")

    def test_every_group_is_registered(self):
        self.assertEqual(
            sorted(catalog.__all__),
            [
                "ASSIGNER", "COMPILE", "DATA", "EFFICIENTNET",
                "GRID_MASK", "MATCH_COST", "PV_MASK",
                "TRAINING",
            ],
        )
        for group_id in EXPECTED_GROUPS:
            with self.subTest(group_id=group_id):
                self.assertIsNotNone(default_registry.get_group(group_id))

    def test_exported_declarations_cover_every_group(self):
        declared = {
            getattr(catalog, name).group_id for name in catalog.__all__
        }
        self.assertEqual(declared, set(EXPECTED_GROUPS))

    def test_targets_point_at_the_official_baseline(self):
        targets = {
            spec.target
            for group_id in EXPECTED_GROUPS
            for replacement_id in default_registry.get_group(group_id).members
            for spec in (default_registry.get_spec(replacement_id),)
            if spec is not None
        }
        self.assertIn(
            "projects.mmdet3d_plugin.bevformer.apis.mmdet_train."
            "custom_train_detector",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.datasets.builder.build_dataloader", targets
        )
        self.assertIn(
            "projects.mmdet3d_plugin.models.utils.grid_mask.GridMask.forward",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.losses.map_loss."
            "OrderedPtsL1Cost.__call__",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_pvmask",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_mask",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "LSSTransform.get_cam_feats",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "normalize_2d_pts",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.assigners.maptr_assigner."
            "MapTRAssigner.assign",
            targets,
        )

    def test_optional_groups_dispatch_through_compat_probes(self):
        for group_id, (condition, member_count) in EXPECTED_CONDITIONS.items():
            group = default_registry.get_group(group_id)
            with self.subTest(group_id=group_id):
                self.assertEqual(len(group.members), member_count)
            for replacement_id in group.members:
                spec = default_registry.get_spec(replacement_id)
                with self.subTest(group_id=group_id, target=spec.target):
                    self.assertEqual(spec.runtime_condition, condition)

    def test_declared_conditions_are_plain_callables(self):
        from turbo_physai.engine.execution.replacements.base import (
            resolve_replacement,
        )

        conditions = {item[0] for item in EXPECTED_CONDITIONS.values()}
        for condition in sorted(conditions):
            probe = resolve_replacement(condition)
            with self.subTest(condition=condition):
                self.assertTrue(callable(probe))
                self.assertFalse(inspect.isclass(probe))
                self.assertIsInstance(probe(), bool)

    def test_catalog_import_does_not_load_torch(self):
        """Declaring groups must not pull Torch into a fresh interpreter."""

        script = (
            "import sys;"
            "import turbo_physai.optimizations.models.maptrv2_optimization.catalog;"
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        environment = dict(os.environ)
        # Ship the parent's import path so ``turbo_physai`` resolves in the child
        # no matter whether it was installed or picked up from the CWD.
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + [entry for entry in sys.path if entry])
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace"),
        )

    def test_mechanism_per_group(self):
        for group_id, mechanism in EXPECTED_GROUPS.items():
            group = default_registry.get_group(group_id)
            for replacement_id in group.members:
                spec = default_registry.get_spec(replacement_id)
                with self.subTest(group_id=group_id):
                    self.assertEqual(spec.mechanism, mechanism)


if __name__ == "__main__":
    unittest.main()
