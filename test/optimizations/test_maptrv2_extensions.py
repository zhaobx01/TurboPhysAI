# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import types

import pytest


torch = pytest.importorskip("torch")


def test_pad_to_static_list_preserves_values_and_mask():
    from turbo_physai.optimizations.models.maptrv2_optimization import assigner

    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded, mask, length = assigner.pad_to_static_list([first])[0]
    assert padded.shape == (assigner.STATIC_GT_COUNT, 2)
    assert length == 2
    torch.testing.assert_close(padded[:2], first)
    assert mask[:2].all()
    assert not mask[2:].any()


def test_static_target_path_uses_padded_gt_contract():
    from turbo_physai.optimizations.models.maptrv2_optimization import assigner

    class Assigner:
        @staticmethod
        def assign(*args):
            del args
            return object(), torch.tensor([[0]])

    class Sampler:
        @staticmethod
        def sample(assign_result, bbox_pred, gt_bboxes):
            del assign_result, bbox_pred, gt_bboxes
            return types.SimpleNamespace(
                pos_inds=torch.tensor([0]),
                neg_inds=torch.tensor([1]),
                pos_assigned_gt_inds=torch.tensor([0]),
                pos_gt_bboxes=torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
            )

    model = types.SimpleNamespace(
        assigner=Assigner(),
        sampler=Sampler(),
        num_classes=3,
    )
    bbox_pred = torch.zeros(2, 4)
    result = assigner.get_target_single(
        model,
        cls_score=torch.zeros(2, 3),
        bbox_pred=bbox_pred,
        pts_pred=torch.zeros(2, 2, 2),
        gt_labels=(torch.tensor([1]), torch.tensor([True]), 1),
        gt_bboxes=(
            torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
            torch.tensor([True]),
            1,
        ),
        gt_shifts_pts=(
            torch.tensor([[[[7.0, 8.0], [9.0, 10.0]]]]),
            torch.tensor([True]),
            1,
        ),
    )
    labels, _, _, _, pts_targets, _, _, _ = result
    assert labels.tolist() == [1, 3]
    torch.testing.assert_close(
        pts_targets[0], torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    )


def test_assign_impl_consumes_padded_cost_mask(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2_optimization import assigner

    module_names = (
        "projects",
        "projects.mmdet3d_plugin",
        "projects.mmdet3d_plugin.maptr",
        "projects.mmdet3d_plugin.maptr.dense_heads",
        "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    for parent, child in zip(module_names, module_names[1:]):
        setattr(modules[parent], child.rsplit(".", 1)[1], modules[child])
    module = modules[module_names[-1]]
    module.normalize_2d_bbox = lambda value, _: value
    module.normalize_2d_pts = lambda value, _: value
    module.normalize_3d_pts = lambda value, _: value
    module.denormalize_2d_bbox = lambda value, _: value
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(
        assigner,
        "_DISABLED_HUNGARIAN",
        lambda cost, *args: (cost.shape, args[-1]),
    )

    class Cost:
        weight = 1.0

        def __call__(self, left, right):
            gt_count = (
                right.shape[0]
                if right.ndim == 1
                else right.shape[0] * right.shape[1]
            )
            return torch.zeros(left.shape[0], gt_count)

    class ScalarCost:
        weight = 1.0

        def __call__(self, left, right):
            del right
            return torch.zeros(left.shape[0], 1)

    class Model:
        cls_cost = Cost()
        reg_cost = ScalarCost()
        iou_cost = ScalarCost()
        pts_cost = Cost()
        z_cfg = {"gt_z_flag": False}
        pc_range = [0.0, 0.0, -1.0, 100.0, 100.0, 1.0]

    _, order_index = assigner._assign_impl(
        Model(),
        torch.zeros(2, 4),
        torch.zeros(2, 3),
        torch.zeros(2, 1, 2),
        (
            torch.zeros(3, 4),
            torch.tensor([True, True, False]),
            2,
        ),
        (torch.zeros(3, dtype=torch.long), torch.tensor([True] * 3), 3),
        (torch.zeros(3, 1, 2, 2), torch.tensor([True] * 3), 3),
    )
    assert order_index.shape == (2, 3)


def test_assigner_is_eager_by_default(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2_optimization import assigner

    captured = []
    monkeypatch.delenv("TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE", raising=False)
    monkeypatch.setattr(
        assigner,
        "_assign_no_grad",
        lambda *args: captured.append(args) or "assignment",
    )
    assert assigner.assign(None, 1, 2, 3, 4, 5, 6, 7, 8) == "assignment"
    assert captured


def test_reference_module_exposes_all_eight_compile_helpers():
    from turbo_physai.optimizations.models.maptrv2_optimization import reference

    assert set(reference._HELPERS) == {
        "matmul_1",
        "matmul_2",
        "matmul_3",
        "extract_metas",
        "down_sample",
        "initialize_queries_and_bev",
        "compute_decoder_predictions",
        "prepare_transformer_inputs",
    }


def test_reference_wrappers_compile_each_helper_once(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2_optimization import reference

    calls = []

    def fake_compile(function, **kwargs):
        calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        reference, "_force_fp32", lambda function, apply_to=None: function
    )
    monkeypatch.setattr(reference, "_COMPILED", {})
    monkeypatch.setattr(reference, "_ACTIVE_MODE", None)
    reference.base_transform_get_geometry_wrapper(
        object(), {"mode": "test-mode"}
    )
    reference.base_transform_forward_wrapper(
        object(), {"mode": "test-mode"}
    )
    assert len(calls) == 8
    assert {kwargs["mode"] for _, kwargs in calls} == {"test-mode"}


def test_reference_matmul_1_keeps_the_coordinate_axis():
    from turbo_physai.optimizations.models.maptrv2_optimization import reference

    batch, cameras = 1, 2
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(
        batch, cameras, -1, -1
    )
    points = torch.tensor(
        [2.0, 3.0, 4.0], dtype=torch.float32
    ).reshape(1, 1, 1, 1, 1, 3).expand(
        batch, cameras, 1, 1, 1, 3
    )
    result = reference.matmul_1(
        None, rotation, points, torch.zeros(batch, cameras, 3)
    )
    assert result.shape == (batch, cameras, 1, 1, 1, 3, 1)
    torch.testing.assert_close(
        result.reshape(batch, cameras, 3),
        torch.tensor([[[8.0, 12.0, 4.0], [8.0, 12.0, 4.0]]]),
    )


def test_reference_down_sample_collapses_depth_to_channels():
    from turbo_physai.optimizations.models.maptrv2_optimization import reference

    model = types.SimpleNamespace(downsample=torch.nn.Identity())
    value = torch.zeros(2, 3, 4, 5, 6)
    result = reference.down_sample(model, value)
    assert result.shape == (2, 18, 5, 4)


def test_reference_bev_pool_accepts_rocm_hw_layout(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2_optimization import reference

    module_names = ("mmdet3d", "mmdet3d.ops")
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["mmdet3d"].ops = modules["mmdet3d.ops"]
    expected = torch.zeros(4, 1, 200, 400, 256)
    modules["mmdet3d.ops"].bev_pool = lambda *args: expected
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    model = types.SimpleNamespace(
        C=256,
        nx=torch.tensor([200, 400, 1]),
        bx=torch.tensor([0.0, 0.0, 0.0]),
        dx=torch.tensor([1.0, 1.0, 1.0]),
    )
    features = torch.zeros(4, 1, 1, 2, 2, 256)
    geometry = torch.ones(4, 1, 1, 2, 2, 3)
    result = reference._bev_pool_5d(model, features, geometry)
    assert result is expected


def test_vectorized_samples_accepts_vertex_sequences(monkeypatch):
    pytest.importorskip("shapely")

    from turbo_physai.optimizations.models.maptrv2_optimization import pv_mask

    module_names = (
        "projects",
        "projects.mmdet3d_plugin",
        "projects.mmdet3d_plugin.datasets",
        "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    for parent, child in zip(module_names, module_names[1:]):
        setattr(modules[parent], child.rsplit(".", 1)[1], modules[child])

    class LiDARInstanceLines:
        def __init__(self, instances, labels, *args, **kwargs):
            del args, kwargs
            self.instances = instances
            self.labels = labels

    modules[module_names[-1]].LiDARInstanceLines = LiDARInstanceLines
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)

    model = types.SimpleNamespace(
        vec_classes=("boundary",),
        CLASS2LABEL={"boundary": 1},
        aux_seg={"use_aux_seg": False},
        sample_dist=1,
        num_samples=20,
        padding=False,
        fixed_num=-1,
        padding_value=-10000,
        patch_size=(100, 100),
    )
    result = pv_mask.gen_vectorized_samples(
        model, {"boundary": [[(0.0, 0.0), (1.0, 1.0)]]}
    )
    assert result["gt_vecs_label"] == [1]
    assert len(result["gt_vecs_pts_loc"].instances) == 1


def test_force_geometric_kernel_extension_is_idempotent(tmp_path):
    from turbo_physai.optimizations.models.maptrv2_optimization import build

    setup_path = (
        tmp_path
        / "projects"
        / "mmdet3d_plugin"
        / "maptr"
        / "modules"
        / "ops"
        / "geometric_kernel_attn"
        / "setup.py"
    )
    setup_path.parent.mkdir(parents=True)
    setup_path.write_text(
        "    if torch.cuda.is_available() and CUDA_HOME is not None:\n",
        encoding="utf-8",
    )
    assert build.force_geometric_kernel_extension(tmp_path) is True
    assert build.force_geometric_kernel_extension(tmp_path) is False
    text = setup_path.read_text(encoding="utf-8")
    assert text.startswith("    if 1:\n")
    assert "# if torch.cuda.is_available()" in text
