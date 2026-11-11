from types import SimpleNamespace

import pytest
import torch

from groot_optimization.zero2_allgather import (
    GroupAllGatherPipeline, regroup_torch_optimizer, scheduler_wrapper, step_wrapper, train_wrapper,
)


def _trainer(optimizer):
    params = [p for group in optimizer.param_groups for p in group["params"]]
    return SimpleNamespace(
        is_deepspeed_enabled=True,
        optimizer=optimizer,
        model=SimpleNamespace(named_parameters=lambda: [(f"p{i}", p) for i, p in enumerate(params)]),
        args=SimpleNamespace(bf16=True, fp16=False, gradient_accumulation_steps=1,
                             save_strategy="steps", save_steps=1000, max_steps=10),
    )


def test_scheduler_tracks_all_regrouped_parameters():
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    optimizer = torch.optim.AdamW(params, lr=0.01, fused=False)
    trainer = _trainer(optimizer)
    def create(self, steps, optimizer=None):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    scheduler = scheduler_wrapper(create, {"group_elements": 2})(trainer, 10, optimizer=optimizer)
    assert len(optimizer.param_groups) == len(scheduler.base_lrs) == 3
    optimizer.step()
    scheduler.step()
    assert [g["lr"] for g in optimizer.param_groups] == [0.005] * 3
    assert len(optimizer._turbo_physai_allgather) == 3


def test_fused_torch_adamw_is_rejected():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], fused=True)
    trainer = _trainer(optimizer)
    with pytest.raises(ValueError, match="fused"):
        scheduler_wrapper(lambda *args, **kwargs: None, {})(trainer, 10, optimizer=optimizer)


def test_apex_selection_precedes_scheduler_regroup(monkeypatch):
    from groot_optimization import zero2_allgather as module
    from groot_optimization.optim import create_optimizer_apex_wrapper

    class FakeFusedAdam(torch.optim.AdamW):
        adam_w_mode = 1
        capturable = False
        master_weights = False

    monkeypatch.setattr(module, "_is_fused_adam_type", lambda kind: kind is FakeFusedAdam)
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    trainer = _trainer(torch.optim.AdamW(params))
    trainer.args.optim = "adamw_torch_fused"
    events = []

    def create_optimizer(self):
        events.append(self.args.optim)
        self.optimizer = FakeFusedAdam(params, lr=0.01)
        return self.optimizer

    def create_scheduler(self, steps, optimizer=None):
        events.append(len(optimizer.param_groups))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)

    optimizer = create_optimizer_apex_wrapper(create_optimizer, {})(trainer)
    scheduler = scheduler_wrapper(create_scheduler, {"group_elements": 2})(trainer, 10, optimizer=optimizer)
    assert events == ["adamw_apex_fused", 2]
    assert trainer.args.optim == "adamw_torch_fused"
    assert len(scheduler.base_lrs) == len(optimizer.param_groups_master) == 2


@pytest.mark.parametrize("keyword", [{"resume_from_checkpoint": "x"}])
def test_train_rejects_resume(keyword):
    trainer = _trainer(torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]))
    def original(self, resume_from_checkpoint=None):
        return "started"
    with pytest.raises(ValueError, match="resume"):
        train_wrapper(original, {})(trainer, **keyword)


def test_pipeline_waits_for_all_work_even_when_one_fails():
    calls = []
    class Work:
        def __init__(self, index): self.index = index
        def wait(self):
            calls.append(self.index)
            if self.index == 0: raise RuntimeError("failed")
    pipeline = GroupAllGatherPipeline(lambda i: Work(i))
    pipeline.submit(0, (object(),))
    pipeline.submit(1, (object(),))
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.wait_all()
    assert calls == [0, 1]
    assert len(pipeline.in_flight_references) == 1


def test_failed_async_work_is_retained_and_prevents_another_step():
    from groot_optimization.zero2_allgather import _finish_allgathers

    class Work:
        def wait(self):
            raise RuntimeError("communication failed")

    pipeline = GroupAllGatherPipeline(lambda _: Work())
    references = (object(),)
    pipeline.submit(0, references)
    optimizer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="communication failed"):
        _finish_allgathers(optimizer, pipeline)
    assert optimizer._turbo_physai_pending_allgather is pipeline
    assert pipeline.in_flight_references == [references]
    with pytest.raises(RuntimeError, match="unfinished"):
        step_wrapper(lambda self, closure=None: None, {})(optimizer)


def test_failed_partial_step_permanently_poisoned(monkeypatch):
    from groot_optimization import zero2_allgather as module

    def fail_after_update(optimizer, closure=None):
        optimizer.updates += 1
        raise RuntimeError("group update failed")

    monkeypatch.setattr(module, "_pipeline_zero2_step", fail_after_update)
    optimizer = SimpleNamespace(updates=0, _turbo_physai_allgather={"activated": True, "calls": 0})
    wrapped = step_wrapper(lambda self, closure=None: None, {})
    with pytest.raises(RuntimeError, match="group update failed"):
        wrapped(optimizer)
    with pytest.raises(RuntimeError, match="previously failed"):
        wrapped(optimizer)
    assert optimizer.updates == 1


@pytest.mark.parametrize("settings", [
    {"bf16": False}, {"gradient_accumulation_steps": 2},
    {"save_strategy": "steps", "save_steps": 2, "max_steps": 10},
])
def test_train_rejects_unsupported_configuration(settings):
    trainer = _trainer(torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]))
    for key, value in settings.items():
        setattr(trainer.args, key, value)
    with pytest.raises(ValueError):
        train_wrapper(lambda self, resume_from_checkpoint=None: None, {})(trainer)


def test_step_rejects_rank_plan_mismatch(monkeypatch):
    from groot_optimization import zero2_allgather as module
    monkeypatch.setattr(module, "_validate_local", lambda obj, world: {"plan": ["rank-0"]})
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    def gather(items, status):
        items[:] = [status, {**status, "record": {"plan": ["rank-1"]}}]
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    wrapped = step_wrapper(lambda self, closure=None: None, {"world_size": 2})
    with pytest.raises(RuntimeError, match="disagree"):
        wrapped(SimpleNamespace())


def test_step_rejects_different_strict_options_across_ranks(monkeypatch):
    from groot_optimization import zero2_allgather as module
    monkeypatch.setattr(module, "_validate_local", lambda obj, world: {"plan": []})
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    def gather(items, status):
        items[:] = [status, {**status, "strict": False}]
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="options disagree"):
        step_wrapper(lambda self, closure=None: None, {"world_size": 2})(SimpleNamespace())


def test_step_strict_false_falls_back_on_validation_failure(monkeypatch):
    from groot_optimization import zero2_allgather as module
    def invalid(*args): raise RuntimeError("unsupported")
    monkeypatch.setattr(module, "_validate_local", invalid)
    monkeypatch.setattr(module.dist, "is_available", lambda: False)
    original = lambda self, closure=None: "fallback"
    optimizer = SimpleNamespace()
    wrapped = step_wrapper(original, {"strict": False})
    assert wrapped(optimizer) == "fallback"
    assert optimizer._turbo_physai_allgather["activated"] is False
