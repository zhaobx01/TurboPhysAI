from types import SimpleNamespace

import pytest
from groot_optimization.optim import create_optimizer_apex_wrapper


def test_optimizer_override_is_temporary_and_respects_offload():
    seen = []
    def original(self):
        seen.append(self.args.optim)
        return "optimizer"
    fn = create_optimizer_apex_wrapper(original, {})
    trainer = SimpleNamespace(args=SimpleNamespace(optim="adamw_torch", deepspeed=None),
                              accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None)))
    assert fn(trainer) == "optimizer"
    assert seen == ["adamw_apex_fused"]
    assert trainer.args.optim == "adamw_torch"
    trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(deepspeed_config={"optimizer": {}})
    fn(trainer)
    assert seen[-1] == "adamw_torch"


def test_optimizer_restores_on_exception():
    def original(self):
        raise ValueError("failure")
    trainer = SimpleNamespace(args=SimpleNamespace(optim="adamw_torch", deepspeed=None),
                              accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None)))
    with pytest.raises(ValueError):
        create_optimizer_apex_wrapper(original, {})(trainer)
    assert trainer.args.optim == "adamw_torch"


def test_transformers_optimizer_enum_is_supported():
    from transformers.training_args import OptimizerNames
    trainer = SimpleNamespace(args=SimpleNamespace(optim=OptimizerNames.ADAMW_TORCH, deepspeed=None),
                              accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None)))
    assert create_optimizer_apex_wrapper(lambda self: self.args.optim, {})(trainer) == "adamw_apex_fused"
    assert trainer.args.optim is OptimizerNames.ADAMW_TORCH
