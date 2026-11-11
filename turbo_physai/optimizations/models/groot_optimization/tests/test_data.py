import copy
import pickle
import socket
from types import SimpleNamespace

import torch
import pytest

from groot_optimization.data import Bf16PixelValuesCollator, trainer_init_bf16_wrapper


class Collator:
    marker = 3
    def __call__(self, features):
        return {"inputs": {"pixel_values": torch.tensor(features, dtype=torch.float32)}}


def test_collator_conversion_pickle_and_forwarded_attributes():
    wrapped = Bf16PixelValuesCollator(Collator())
    for obj in (wrapped, copy.copy(wrapped), pickle.loads(pickle.dumps(wrapped))):
        assert obj.marker == 3
        assert torch.equal(obj([1.25])["inputs"]["pixel_values"], torch.tensor([1.25], dtype=torch.bfloat16))


def test_trainer_wraps_only_deepspeed_bf16():
    def init(self, args=None, data_collator=None):
        self.data_collator = data_collator
    init = trainer_init_bf16_wrapper(init, {})
    trainer = SimpleNamespace()
    collator = Collator()
    init(trainer, SimpleNamespace(deepspeed="ds.json", bf16=True, fp16=False), collator)
    assert isinstance(trainer.data_collator, Bf16PixelValuesCollator)
    init(trainer, SimpleNamespace(deepspeed=None, bf16=True, fp16=False), collator)
    assert trainer.data_collator is collator


def test_custom_gr00t_trainer_keywords_are_preserved():
    def init(self, *args, **kwargs):
        self.kwargs = kwargs
    trainer = SimpleNamespace()
    args = SimpleNamespace(deepspeed="ds", bf16=True, fp16=False)
    trainer_init_bf16_wrapper(init, {})(trainer, args=args, data_collator=Collator(),
                                       multiprocessing_context="fork", action_offset=5)
    assert trainer.kwargs["multiprocessing_context"] == "fork"
    assert trainer.kwargs["action_offset"] == 5
    assert isinstance(trainer.kwargs["data_collator"], Bf16PixelValuesCollator)


def test_collator_runs_in_dataloader_worker():
    try:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except PermissionError:
        pytest.skip("sandbox forbids the Unix socket needed for tensor IPC")
    loader = torch.utils.data.DataLoader([1.25, 2.5], batch_size=2, num_workers=1,
                                        multiprocessing_context="spawn", timeout=30,
                                        collate_fn=Bf16PixelValuesCollator(Collator()))
    assert next(iter(loader))["inputs"]["pixel_values"].dtype == torch.bfloat16
