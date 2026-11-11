import pytest
import torch
from types import SimpleNamespace

from groot_optimization.kv_cache import disable_kv_cache_wrapper


@pytest.fixture
def text_model():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    config = Qwen3VLTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    return Qwen3VLTextModel(config).train()


def test_default_cache_disabled_without_changing_hidden_states(text_model):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    tokens = torch.tensor([[1, 2, 3]])
    original = Qwen3VLTextModel.forward
    baseline = original(text_model, input_ids=tokens, output_hidden_states=True)
    assert text_model.config.use_cache is True
    assert baseline.past_key_values is not None
    output = disable_kv_cache_wrapper(original, {})(text_model, input_ids=tokens, output_hidden_states=True)
    assert output.past_key_values is None
    assert torch.equal(output.last_hidden_state, baseline.last_hidden_state)
    assert all(torch.equal(a, b) for a, b in zip(output.hidden_states, baseline.hidden_states))


def test_explicit_cache_and_existing_cache_are_not_overridden(text_model):
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    fn = disable_kv_cache_wrapper(Qwen3VLTextModel.forward, {})
    tokens = torch.tensor([[1, 2]])
    assert fn(text_model, input_ids=tokens, use_cache=True).past_key_values is not None
    cache = DynamicCache(config=text_model.config)
    assert fn(text_model, input_ids=tokens, past_key_values=cache).past_key_values is cache


def test_positional_explicit_use_cache_is_respected():
    def original(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=None):
        return use_cache, past_key_values
    wrapped = disable_kv_cache_wrapper(original, {})
    assert wrapped(SimpleNamespace()) == (False, None)
    assert wrapped(SimpleNamespace(), None, None, None, True) == (True, None)
    assert wrapped(SimpleNamespace(), use_cache=None) == (None, None)
