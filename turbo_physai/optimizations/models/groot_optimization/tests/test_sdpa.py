import contextlib

import pytest
import torch

from groot_optimization.catalog import DIT_HIP_MATH_SDPA
from groot_optimization.replacements import dit_sdpa_context_wrapper
from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.execution.replacements import default_handlers


def test_group_targets_groot_dit_context():
    (spec,) = DIT_HIP_MATH_SDPA.specs
    assert spec.mechanism is Mechanism.WRAPPER
    assert spec.target == "gr00t.model.modules.dit._sdpa_context"


@pytest.mark.parametrize("mode", [None, "math"])
def test_hip_forces_math_temporarily_and_restores_backend_flags(monkeypatch, mode):
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    if mode is None:
        monkeypatch.delenv("GR00T_DIT_SDPA_MODE", raising=False)
    else:
        monkeypatch.setenv("GR00T_DIT_SDPA_MODE", mode)
    flags = lambda: (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
    before = flags()
    wrapped = dit_sdpa_context_wrapper(contextlib.nullcontext, {})
    with wrapped():
        assert flags() == (True, False, False)
    assert flags() == before


@pytest.mark.parametrize("hip,mode", [(None, None), ("test-hip", "default")])
def test_other_platform_or_default_override_uses_original(monkeypatch, hip, mode):
    monkeypatch.setattr(torch.version, "hip", hip)
    if mode is None:
        monkeypatch.delenv("GR00T_DIT_SDPA_MODE", raising=False)
    else:
        monkeypatch.setenv("GR00T_DIT_SDPA_MODE", mode)
    sentinel = object()
    wrapped = dit_sdpa_context_wrapper(lambda: sentinel, {})
    assert wrapped() is sentinel


def test_real_dit_replacement_is_applied_and_restored(monkeypatch):
    dit = pytest.importorskip("gr00t.model.modules.dit")
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.delenv("GR00T_DIT_SDPA_MODE", raising=False)
    original = dit._sdpa_context
    handler = default_handlers()[Mechanism.WRAPPER]
    prepared = handler.prepare(DIT_HIP_MATH_SDPA.specs[0], {})
    snapshot = handler.snapshot(prepared)
    try:
        handler.apply(prepared)
        with dit._sdpa_context():
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.flash_sdp_enabled()
    finally:
        handler.restore(snapshot)
    assert dit._sdpa_context is original
