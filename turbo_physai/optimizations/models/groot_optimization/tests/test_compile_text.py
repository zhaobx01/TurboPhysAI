import copy

import pytest
import torch

from groot_optimization.compile import compile_method_wrapper
from groot_optimization.compile_text import compile_text_wrapper
from groot_optimization.kv_cache import disable_kv_cache_wrapper


def test_text_compile_preserves_outputs_and_gradients(monkeypatch):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextDecoderLayer

    cfg = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                            head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    baseline = Qwen3VLTextModel(cfg).train()
    actual = copy.deepcopy(baseline)
    inputs = torch.tensor([[1, 2, 3]])
    expected = baseline(input_ids=inputs, use_cache=False, output_hidden_states=True)
    expected.last_hidden_state.square().sum().backward()
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda f, **kwargs: original_compile(f, backend="eager"))
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compile_method_wrapper(Qwen3VLTextDecoderLayer.forward, {}))
    result = actual(input_ids=inputs, use_cache=False, output_hidden_states=True)
    torch.testing.assert_close(result.last_hidden_state, expected.last_hidden_state)
    assert len(result.hidden_states) == len(expected.hidden_states)
    for result_hidden, expected_hidden in zip(result.hidden_states, expected.hidden_states):
        torch.testing.assert_close(result_hidden, expected_hidden)
    result.last_hidden_state.square().sum().backward()
    for a, b in zip(actual.parameters(), baseline.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad)


def test_text_compile_dependencies():
    from groot_optimization.catalog import COMPILE_TEXT
    assert set(COMPILE_TEXT.definition.depends_on) == {
        "groot.qwen.flash_attn_dynamo_disable", "groot.qwen.disable_kv_cache"
    }


def test_text_compile_records_layer_calls_and_respects_disable(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    def forward(self, x): return x + 1
    wrapped = compile_text_wrapper(forward, {})
    layer = SimpleNamespace(layer_idx=3)
    assert wrapped(layer, 1) == 2
    assert wrapped(layer, 2) == 3
    assert layer._turbo_physai_text_compile_calls == 2
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert compile_text_wrapper(forward, {}) is forward


def test_sixteen_text_layers_without_cache_share_compiled_graph(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    model = Qwen3VLTextModel(config).eval()
    graph_count = []
    def backend(graph, inputs):
        graph_count.append(graph)
        return graph.forward
    torch._dynamo.reset()
    compiled = torch.compile(Qwen3VLTextDecoderLayer.forward, backend=backend)
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compiled)
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    assert len(graph_count) == 1


def test_sixteen_text_layers_with_cache_hit_recompile_limit(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    model = Qwen3VLTextModel(config).eval()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", torch.compile(Qwen3VLTextDecoderLayer.forward, backend=backend))
    with torch._dynamo.config.patch(recompile_limit=8, fail_on_recompile_limit_hit=True), torch.no_grad():
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit, match="recompile_limit"):
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    assert len(graphs) == 8


def test_text_compile_with_gradient_checkpointing(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel
    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    model = Qwen3VLTextModel(config).train()
    model.gradient_checkpointing_enable()
    compiler = torch.compile
    monkeypatch.setattr(torch, "compile", lambda f, **kwargs: compiler(f, backend="eager"))
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compile_method_wrapper(Qwen3VLTextDecoderLayer.forward, {}))
    output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False, output_hidden_states=True)
    output.last_hidden_state.square().sum().backward()
    assert len(output.hidden_states) == 3
    assert model.embed_tokens.weight.grad is not None


@pytest.mark.hcu
@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip, reason="requires an HCU")
def test_text_inductor_on_hcu_matches_eager_outputs_and_gradients(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    expected = Qwen3VLTextModel(config).to("cuda").train()
    actual = copy.deepcopy(expected)
    tokens = torch.tensor([[1, 2, 3]], device="cuda")
    output = expected(input_ids=tokens, use_cache=False, output_hidden_states=True)
    output.last_hidden_state.square().sum().backward()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compile_text_wrapper(Qwen3VLTextDecoderLayer.forward, {}))
    compiled = actual(input_ids=tokens, use_cache=False, output_hidden_states=True)
    compiled.last_hidden_state.square().sum().backward()
    for actual_hidden, expected_hidden in zip(compiled.hidden_states, output.hidden_states):
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-3, atol=1e-3)
    for actual_param, expected_param in zip(actual.parameters(), expected.parameters()):
        if actual_param.grad is None or expected_param.grad is None:
            assert actual_param.grad is None and expected_param.grad is None
        else:
            torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-3, atol=1e-3)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
