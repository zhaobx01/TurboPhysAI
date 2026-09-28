# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the MapTRv2 static-graph DDP settings.

The reference implementation passes ``static_graph=True`` and
``find_unused_parameters=False`` from ``mmdet_train.py``, driven by a config
field the official baseline does not have. The Group wraps the constructor of
``mmcv.parallel.distributed.MMDistributedDataParallel`` instead, which must not
disturb the Torch DDP constructor it inherits from.
"""

from __future__ import annotations

import importlib

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("mmcv")


MODULE = "turbo_physai.optimizations.models.maptrv2_optimization.ddp"
TARGET = "mmcv.parallel.distributed.MMDistributedDataParallel.__init__"

_STATIC_GRAPH_ENV = "TURBO_PHYSAI_DDP_STATIC_GRAPH"
_FIND_UNUSED_ENV = "TURBO_PHYSAI_DDP_FIND_UNUSED_PARAMETERS"


@pytest.fixture
def ddp(monkeypatch):
    monkeypatch.delenv(_STATIC_GRAPH_ENV, raising=False)
    monkeypatch.delenv(_FIND_UNUSED_ENV, raising=False)
    return importlib.import_module(MODULE)


def _install(ddp, monkeypatch, options=None, **env):
    """Install the wrapper over a recorder and return the recorded call."""

    for name, value in env.items():
        monkeypatch.setenv(name, value)

    recorded = {}

    def original(self, *args, **kwargs):
        recorded["self"] = self
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return "constructed"

    wrapper = ddp.ddp_constructor_wrapper(original, options or {})
    recorded["wrapper"] = wrapper
    return recorded


def test_default_injects_the_reference_settings(ddp, monkeypatch):
    recorded = _install(ddp, monkeypatch)
    result = recorded["wrapper"]("model", device_ids=[0], broadcast_buffers=False)

    assert result == "constructed"
    assert recorded["kwargs"]["static_graph"] is True
    assert recorded["kwargs"]["find_unused_parameters"] is False


def test_static_graph_can_be_disabled(ddp, monkeypatch):
    recorded = _install(ddp, monkeypatch, **{_STATIC_GRAPH_ENV: "0"})
    recorded["wrapper"]("model", find_unused_parameters=True)

    assert "static_graph" not in recorded["kwargs"]
    # The caller's baseline value is still overridden unless the second
    # switch is turned off as well.
    assert recorded["kwargs"]["find_unused_parameters"] is False


def test_find_unused_parameters_can_be_kept(ddp, monkeypatch):
    recorded = _install(ddp, monkeypatch, **{_FIND_UNUSED_ENV: "0"})
    recorded["wrapper"]("model", find_unused_parameters=True)

    assert recorded["kwargs"]["static_graph"] is True
    assert recorded["kwargs"]["find_unused_parameters"] is True


def test_both_switches_off_leaves_the_call_untouched(ddp, monkeypatch):
    recorded = _install(
        ddp, monkeypatch, **{_STATIC_GRAPH_ENV: "0", _FIND_UNUSED_ENV: "0"}
    )
    recorded["wrapper"]("model", find_unused_parameters=True)

    assert recorded["kwargs"] == {"find_unused_parameters": True}


def test_options_override_the_environment(ddp, monkeypatch):
    recorded = _install(
        ddp,
        monkeypatch,
        options={"static_graph": False, "find_unused_parameters": False},
        **{_STATIC_GRAPH_ENV: "1", _FIND_UNUSED_ENV: "1"},
    )
    recorded["wrapper"]("model", find_unused_parameters=True)

    assert "static_graph" not in recorded["kwargs"]
    assert recorded["kwargs"]["find_unused_parameters"] is True


def test_caller_supplied_static_graph_is_preserved(ddp, monkeypatch):
    recorded = _install(ddp, monkeypatch)
    recorded["wrapper"]("model", static_graph=False)

    assert recorded["kwargs"]["static_graph"] is False


def test_positional_arguments_are_forwarded(ddp, monkeypatch):
    recorded = _install(ddp, monkeypatch)
    recorded["wrapper"]("model", [0], None, 0)

    assert recorded["self"] == "model"
    assert recorded["args"] == ([0], None, 0)


def test_wrapper_keeps_the_original_metadata(ddp):
    def original(self, *args, **kwargs):  # pragma: no cover - metadata only
        return None

    wrapper = ddp.ddp_constructor_wrapper(original, {})

    assert wrapper.__name__ == original.__name__
    assert wrapper.__wrapped__ is original


def test_install_shadows_only_the_mmcv_subclass(ddp):
    """The Torch constructor the subclass inherits from must stay intact."""

    from mmcv.parallel.distributed import MMDistributedDataParallel
    from torch.nn.parallel.distributed import DistributedDataParallel

    from turbo_physai.engine.execution.replacements.base import (
        resolve_attribute,
        set_attribute,
    )

    resolved = resolve_attribute(TARGET)
    assert resolved.parent is MMDistributedDataParallel
    assert resolved.original is DistributedDataParallel.__init__
    assert MMDistributedDataParallel.__init__ is DistributedDataParallel.__init__

    wrapper = ddp.ddp_constructor_wrapper(resolved.original, {})
    set_attribute(resolved, wrapper)
    try:
        assert MMDistributedDataParallel.__init__ is wrapper
        assert DistributedDataParallel.__init__ is resolved.original
    finally:
        del MMDistributedDataParallel.__init__

    assert MMDistributedDataParallel.__init__ is DistributedDataParallel.__init__
