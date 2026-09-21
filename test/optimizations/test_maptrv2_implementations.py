# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical equivalence tests for the MapTRv2 replacement implementations."""

import os
import sys
import types
import unittest
import unittest.mock

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for catalog checks
    torch = None

try:
    import mmcv  # noqa: F401
except Exception:  # pragma: no cover - the model environment ships mmcv
    mmcv = None

try:
    import cv2
except Exception:  # pragma: no cover - opencv ships with the model stack
    cv2 = None

try:
    from shapely.geometry import LineString
except Exception:  # pragma: no cover - shapely ships with the model stack
    LineString = None


class _GridMaskStub:
    """Minimal stand-in for ``GridMask`` so the math can be tested alone."""

    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5,
                 mode=0, prob=1.0):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.l = None
        self.training = True


def _official_mask(mask, use_h, use_w, mode, distance, start_h, start_w,
                   keep, expanded_h, expanded_w):
    """Official numpy mask construction (rotate == 1 is the identity)."""

    if use_h:
        for index in range(expanded_h // distance):
            start = distance * index + start_h
            mask[start:min(start + keep, expanded_h), :] *= 0
    if use_w:
        for index in range(expanded_w // distance):
            start = distance * index + start_w
            mask[:, start:min(start + keep, expanded_w)] *= 0
    if mode == 1:
        mask = 1 - mask
    return mask


def _official_pv_mask(line_ego, proj_mat, shape, color=1, thickness=1,
                      z=-1.6):
    """Baseline ``VectorizedLocalMap.line_ego_to_pvmask`` body, inlined."""

    distances = np.linspace(0, line_ego.length, 200)
    coords = np.array(
        [list(line_ego.interpolate(distance).coords) for distance in distances]
    ).reshape(-1, 2)
    pts_num = coords.shape[0]
    zeros = np.zeros((pts_num, 1))
    zeros[:] = z
    ones = np.ones((pts_num, 1))
    lidar_coords = np.concatenate([coords, zeros, ones], axis=1).transpose(1, 0)
    pix_coords = proj_mat @ lidar_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(mask, np.int32([pix_coords]), False, color=color,
                  thickness=thickness)
    return mask


def _official_line_ego_to_mask(canvas_size, scale_x, scale_y, line_ego,
                               color=1, thickness=3):
    """Baseline ``VectorizedLocalMap.line_ego_to_mask`` body, inlined."""

    from shapely import affinity

    trans_x = canvas_size[1] / 2
    trans_y = canvas_size[0] / 2
    line_ego = affinity.scale(line_ego, scale_x, scale_y, origin=(0, 0))
    line_ego = affinity.affine_transform(
        line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
    coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
    coords = coords.reshape((-1, 2))
    assert len(coords) >= 2
    mask = np.zeros(canvas_size, dtype=np.uint8)
    cv2.polylines(mask, np.int32([coords]), False, color=color,
                  thickness=thickness)
    return mask


@unittest.skipIf(torch is None or mmcv is None,
                 "torch and mmcv are required for GridMask tests")
class GridMaskTest(unittest.TestCase):
    def _run(self, module, stub, x, distance, start_h, start_w, prob_draw):
        original_rand = np.random.rand
        original_randint = np.random.randint
        draws = {"start": iter([start_h, start_w])}

        def fake_rand(*args):
            if args:
                return np.full(args, 0.25, dtype=np.float32)
            return prob_draw

        def fake_randint(*args):
            if args == (2, x.size(2)):
                return distance
            if args == (stub.rotate,):
                return 0
            if args == (distance,):
                return next(draws["start"])
            raise AssertionError(f"unexpected randint args: {args}")

        np.random.rand = fake_rand
        np.random.randint = fake_randint
        try:
            return module.grid_mask_forward(stub, x)
        finally:
            np.random.rand = original_rand
            np.random.randint = original_randint

    def _expected(self, stub, x, distance, start_h, start_w):
        _, _, height, width = x.shape
        expanded_h, expanded_w = int(1.5 * height), int(1.5 * width)
        keep = min(max(int(distance * stub.ratio + 0.5), 1), distance - 1)
        mask = np.ones((expanded_h, expanded_w), np.float32)
        mask = _official_mask(mask, stub.use_h, stub.use_w, stub.mode, distance,
                              start_h, start_w, keep, expanded_h, expanded_w)
        mask = mask[(expanded_h - height) // 2:(expanded_h - height) // 2 + height,
                    (expanded_w - width) // 2:(expanded_w - width) // 2 + width]
        return torch.from_numpy(mask)

    def test_mask_matches_official_numpy_construction(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import grid_mask

        stub = _GridMaskStub(True, True, rotate=1, offset=False, ratio=0.5,
                             mode=1, prob=0.7)
        x = torch.arange(2 * 3 * 32 * 40, dtype=torch.float32).reshape(
            2, 3, 32, 40).contiguous()
        expected_mask = self._expected(stub, x, distance=6, start_h=2,
                                       start_w=4)
        actual = self._run(grid_mask, stub, x, 6, 2, 4, prob_draw=0.0)
        expected = x.view(-1, 32, 40) * expected_mask
        torch.testing.assert_close(
            actual.view(-1, 32, 40), expected, rtol=0, atol=0)

    def test_offset_path_matches_official_numpy_construction(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import grid_mask

        stub = _GridMaskStub(True, True, rotate=1, offset=True, ratio=0.5,
                             mode=0, prob=0.7)
        x = torch.arange(1 * 2 * 16 * 24, dtype=torch.float32).reshape(
            1, 2, 16, 24).contiguous()
        expected_mask = self._expected(stub, x, distance=4, start_h=1,
                                       start_w=3)
        actual = self._run(grid_mask, stub, x, 4, 1, 3, prob_draw=0.0)
        offset = torch.full((16, 24), 2 * (0.25 - 0.5), dtype=torch.float32)
        expected = (x.view(-1, 16, 24) * expected_mask
                    + offset * (1 - expected_mask))
        torch.testing.assert_close(
            actual.view(-1, 16, 24), expected, rtol=0, atol=0)

    def test_probability_shortcut_returns_input(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import grid_mask

        stub = _GridMaskStub(True, True, prob=0.7)
        x = torch.ones(1, 1, 8, 8)
        self.assertIs(self._run(grid_mask, stub, x, 3, 0, 0, prob_draw=0.9), x)

    def test_eval_mode_returns_input(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import grid_mask

        stub = _GridMaskStub(True, True, prob=0.7)
        stub.training = False
        x = torch.ones(1, 1, 8, 8)
        self.assertIs(grid_mask.grid_mask_forward(stub, x), x)


@unittest.skipIf(torch is None, "torch is required for implementation tests")
class MatchCostTest(unittest.TestCase):
    def test_broadcast_cost_equals_cdist(self):
        from turbo_physai.optimizations.models.maptrv2_optimization.match_cost import ordered_pts_l1_cost_call

        class _Cost:
            weight = 0.5

        generator = torch.Generator().manual_seed(0)
        # float64 first: the two spellings are algebraically the same sum of
        # absolute differences, so only the reduction order may differ.
        bbox64 = torch.rand(100, 20, 2, generator=generator,
                            dtype=torch.float64)
        gt64 = torch.rand(7, 2, 20, 2, generator=generator,
                          dtype=torch.float64)
        expected = torch.cdist(
            bbox64.view(bbox64.size(0), -1),
            gt64.flatten(2).view(7 * 2, -1), p=1) * _Cost.weight
        actual = ordered_pts_l1_cost_call(_Cost(), bbox64, gt64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

        # float32 (the training dtype) may only differ by accumulated rounding.
        bbox32 = torch.rand(100, 20, 2, generator=generator)
        gt32 = torch.rand(7, 2, 20, 2, generator=generator)
        expected32 = torch.cdist(
            bbox32.view(bbox32.size(0), -1),
            gt32.flatten(2).view(7 * 2, -1), p=1) * _Cost.weight
        actual32 = ordered_pts_l1_cost_call(_Cost(), bbox32, gt32)
        torch.testing.assert_close(actual32, expected32, rtol=1e-5, atol=1e-5)


class TrainingFlagsTest(unittest.TestCase):
    def test_option_flag_prefers_explicit_options(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import training

        self.assertFalse(training._option_flag(
            {"channels_last": False}, "channels_last", "NO_SUCH_ENV", True))
        self.assertTrue(training._option_flag(
            {"channels_last": True}, "channels_last", "NO_SUCH_ENV", False))

    def test_option_flag_falls_back_to_default(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import training

        self.assertTrue(training._option_flag(
            {}, "channels_last", "NO_SUCH_ENV", True))

    def test_channels_last_scope_defaults_to_model(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import training

        with unittest.mock.patch.dict(os.environ):
            os.environ.pop("TURBO_PHYSAI_CHANNELS_LAST_SCOPE", None)
            self.assertEqual(training._channels_last_scope({}), "model")
            self.assertEqual(
                training._channels_last_scope(
                    {"channels_last_scope": "BACKBONE"}),
                "backbone")
            with self.assertRaises(ValueError):
                training._channels_last_scope(
                    {"channels_last_scope": "layer"})

    def test_matmul_precision_defaults_to_reference_value(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import training

        with unittest.mock.patch.dict(os.environ):
            os.environ.pop("TURBO_PHYSAI_MATMUL_PRECISION", None)
            self.assertEqual(training._matmul_precision({}), "high")
            self.assertEqual(
                training._matmul_precision({"matmul_precision": "OFF"}),
                "off")


class CompatProbeTest(unittest.TestCase):
    """The probes back the ``runtime_condition`` of the optional Groups."""

    @staticmethod
    def _stub_module(name, **attributes):
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        return module

    def test_compile_probe_follows_torch_capability(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import compat

        compiled = self._stub_module("torch", compile=lambda target, **_: target)
        with unittest.mock.patch.dict(sys.modules, {"torch": compiled}):
            self.assertTrue(compat.torch_compile_available())
        with unittest.mock.patch.dict(
                sys.modules, {"torch": self._stub_module("torch")}):
            self.assertFalse(compat.torch_compile_available())

    def test_compile_probe_honours_the_kill_switch(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import compat

        compiled = self._stub_module("torch", compile=lambda target, **_: target)
        with unittest.mock.patch.dict(sys.modules, {"torch": compiled}):
            with unittest.mock.patch.dict(
                    os.environ, {"TURBO_PHYSAI_DISABLE_TORCH_COMPILE": "1"}):
                self.assertFalse(compat.torch_compile_available())

    def test_dynamo_probe_checks_dynamo_disable(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import compat

        dynamo = self._stub_module("torch._dynamo", disable=lambda item: item)
        with unittest.mock.patch.dict(
                sys.modules, {"torch": self._stub_module("torch", _dynamo=dynamo)}):
            self.assertTrue(compat.dynamo_available())
        with unittest.mock.patch.dict(
                sys.modules, {"torch": self._stub_module("torch")}):
            self.assertFalse(compat.dynamo_available())

    def test_probes_tolerate_a_missing_torch(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import compat

        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            self.assertFalse(compat.torch_compile_available())
            self.assertFalse(compat.dynamo_available())

    def test_probes_accept_the_wrapped_call_arguments(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import compat

        # The engine calls a condition as ``condition(*args, **kwargs)``.
        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            self.assertIsInstance(
                compat.torch_compile_available(object(), key=1), bool)
            self.assertIsInstance(
                compat.dynamo_available(object(), key=1), bool)


_PROJECTION = np.array([
    [200.0, 0.0, 0.0, 100.0],
    [0.0, 10.0, 0.0, 40.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
])


@unittest.skipIf(LineString is None, "shapely is required for the sampling test")
class PvMaskSamplingTest(unittest.TestCase):
    def test_resampling_matches_shapely_interpolation(self):
        from turbo_physai.optimizations.models.maptrv2_optimization.pv_mask import sample_line_arc_length

        line = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 8.0)])
        distances = np.linspace(0.0, line.length, 200)
        expected = np.array(
            [list(line.interpolate(distance).coords) for distance in distances]
        ).reshape(-1, 2)
        np.testing.assert_allclose(
            sample_line_arc_length(line), expected, rtol=0.0, atol=1e-9)

    def test_resampling_uses_equal_arc_length_steps(self):
        from turbo_physai.optimizations.models.maptrv2_optimization.pv_mask import sample_line_arc_length

        line = LineString([(0.0, 0.0), (3.0, 4.0)])
        np.testing.assert_allclose(
            sample_line_arc_length(line, n_points=5),
            [[0.0, 0.0], [0.75, 1.0], [1.5, 2.0], [2.25, 3.0], [3.0, 4.0]],
            rtol=0.0,
            atol=1e-12,
        )


class PvMaskProjectionTest(unittest.TestCase):
    def test_projection_matches_the_perspective_helper(self):
        from turbo_physai.optimizations.models.maptrv2_optimization.pv_mask import project_points

        coords = np.array([[0.0, 0.0], [1.0, -2.0], [4.0, 3.0]])
        zeros = np.full((coords.shape[0], 1), -1.6)
        ones = np.ones((coords.shape[0], 1))
        lidar_coords = np.concatenate(
            [coords, zeros, ones], axis=1).transpose(1, 0)
        pix_coords = _PROJECTION @ lidar_coords
        valid = pix_coords[2, :] > 0
        expected = (pix_coords[:2, valid]
                    / (pix_coords[2, valid] + 1e-7)).transpose(1, 0)
        np.testing.assert_allclose(
            project_points(coords, _PROJECTION, z=-1.6),
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_points_behind_the_camera_are_dropped(self):
        from turbo_physai.optimizations.models.maptrv2_optimization.pv_mask import project_points

        coords = np.array([[1.0, 1.0], [-1.0, 1.0]])
        visible = project_points(coords, _PROJECTION, z=0.0)
        behind = project_points(coords, _PROJECTION, z=-3.0)
        self.assertEqual(visible.shape, (2, 2))
        # ``perspective`` divides by ``z + 1e-7`` and both points project to
        # ``z = 3``, so the exact expectation carries the same epsilon factor.
        expected = np.array(
            [[100.0, 50.0 / 3.0], [-100.0 / 3.0, 50.0 / 3.0]]
        ) * (3.0 / (3.0 + 1e-7))
        np.testing.assert_allclose(
            visible,
            expected,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(behind.shape, (0, 2))


@unittest.skipIf(LineString is None or cv2 is None,
                 "shapely and opencv are required for the drawing test")
class PvMaskDrawingTest(unittest.TestCase):
    def test_vectorized_drawing_reproduces_the_official_loop(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import pv_mask

        line = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 8.0)])
        shape = (180, 320)
        expected = _official_pv_mask(line, _PROJECTION, shape, thickness=3)
        actual = np.zeros(shape, dtype=np.uint8)
        pv_mask.line_ego_to_pvmask(None, line, actual, _PROJECTION, thickness=3)
        np.testing.assert_array_equal(actual, expected)

    def test_semantic_mask_reproduces_the_official_two_step_transform(self):
        from turbo_physai.optimizations.models.maptrv2_optimization import (
            pv_mask,
        )

        canvas_size = (200, 100)
        scale_x = canvas_size[1] / 100.0
        scale_y = canvas_size[0] / 200.0
        line = LineString([(-12.0, 3.5), (4.0, -9.0), (18.0, 21.0)])
        container = types.SimpleNamespace(
            canvas_size=canvas_size, scale_x=scale_x, scale_y=scale_y)
        for thickness in (1, 3, 5):
            with self.subTest(thickness=thickness):
                expected = _official_line_ego_to_mask(
                    canvas_size, scale_x, scale_y, line, thickness=thickness)
                actual = np.zeros(canvas_size, dtype=np.uint8)
                pv_mask.line_ego_to_mask(
                    container, line, actual, thickness=thickness)
                np.testing.assert_array_equal(actual, expected)

    def test_scale_translate_geom_matches_two_step_affinity(self):
        from shapely import affinity

        from turbo_physai.optimizations.models.maptrv2_optimization.pv_mask import (
            scale_translate_geom,
        )

        line = LineString([(0.0, 0.0), (10.0, -3.0), (12.5, 7.25)])
        for origin in ((0.0, 0.0), (3.5, -2.25), (-11.0, 4.0)):
            with self.subTest(origin=origin):
                two_step = affinity.affine_transform(
                    affinity.scale(line, 2.5, 0.75, origin=origin),
                    [1.0, 0.0, 0.0, 1.0, 7.0, -9.0])
                one_step = scale_translate_geom(
                    line, 2.5, 0.75, 7.0, -9.0, origin=origin)
                np.testing.assert_array_equal(
                    np.asarray(one_step.coords), np.asarray(two_step.coords))


if __name__ == "__main__":
    unittest.main()
