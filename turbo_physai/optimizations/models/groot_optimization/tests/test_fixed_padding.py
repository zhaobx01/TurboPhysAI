from types import SimpleNamespace

import pytest
import torch

from groot_optimization.fixed_padding import fixed_padding_wrapper
from groot_optimization.data import Bf16PixelValuesCollator


def trainer(pad=7):
    return SimpleNamespace(data_collator=SimpleNamespace(
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=pad))))


def test_pad_156_left_pads_before_original_without_mutating_input():
    seen = []
    def original(self, inputs):
        seen.append(inputs)
        return inputs
    wrapped = fixed_padding_wrapper(original, {})
    ids = torch.tensor([[7, 4, 5], [6, 8, 9]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    batch = {"inputs": {"input_ids": ids, "attention_mask": mask, "state": torch.ones(2, 3)}}
    result = wrapped(trainer(), batch)
    values = result["inputs"]
    assert values["input_ids"].shape == (2, 156)
    assert torch.all(values["input_ids"][:, :153] == 7)
    assert torch.equal(values["input_ids"][:, -3:], ids)
    assert torch.all(values["attention_mask"][:, :153] == 0)
    assert torch.equal(values["attention_mask"][:, -3:], mask)
    assert values["state"] is batch["inputs"]["state"]
    assert batch["inputs"]["input_ids"] is ids
    assert seen == [result]


@pytest.mark.parametrize("width", [156, 157])
def test_exact_limit_and_oversized_input(width):
    fn = fixed_padding_wrapper(lambda self, x: x, {})
    batch = {"inputs": {"input_ids": torch.ones((1, width), dtype=torch.long),
                        "attention_mask": torch.ones((1, width), dtype=torch.long)}}
    if width == 156:
        assert fn(trainer(), batch)["inputs"]["input_ids"].shape == (1, 156)
    else:
        with pytest.raises(ValueError, match="156"):
            fn(trainer(), batch)


@pytest.mark.parametrize("mask", [[[1, 0, 1]], [[0, 2, 1]]])
def test_reject_invalid_attention_mask(mask):
    batch = {"inputs": {"input_ids": torch.ones((1, 3), dtype=torch.long),
                        "attention_mask": torch.tensor(mask)}}
    with pytest.raises(ValueError, match="left-padded"):
        fixed_padding_wrapper(lambda self, x: x, {})(trainer(), batch)


def test_reject_unknown_fields_and_missing_tokenizer():
    batch = {"inputs": {"input_ids": torch.ones((1, 2), dtype=torch.long),
                        "attention_mask": torch.ones((1, 2), dtype=torch.long)}}
    fn = fixed_padding_wrapper(lambda self, x: x, {})
    with pytest.raises(ValueError, match="pad token"):
        fn(trainer(None), batch)
    batch["inputs"]["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        fn(trainer(), batch)


def test_pad_token_resolution_through_bf16_collator():
    instance = trainer()
    instance.data_collator = Bf16PixelValuesCollator(instance.data_collator)
    batch = {"inputs": {"input_ids": torch.tensor([[4]]), "attention_mask": torch.tensor([[1]])}}
    result = fixed_padding_wrapper(lambda self, x: x, {})(instance, batch)
    assert result["inputs"]["input_ids"][0, 0].item() == 7
