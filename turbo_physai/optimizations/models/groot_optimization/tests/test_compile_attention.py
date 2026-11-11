import torch

from groot_optimization.compile import compile_method_wrapper
from groot_optimization.attention import flash_attn_dynamo_disable_wrapper


def test_compile_mode_and_disable(monkeypatch):
    calls = []
    def compile_fn(fn, **kwargs):
        calls.append(kwargs)
        return fn
    monkeypatch.setattr(torch, "compile", compile_fn)
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    def f(x): return x + 1
    assert compile_method_wrapper(f, {"mode": "default"})(2) == 3
    assert calls == [{"mode": "default"}]
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert compile_method_wrapper(f, {}) is f


def test_attention_wrapper_is_disabled_for_dynamo():
    def f(x): return x + 1
    wrapped = flash_attn_dynamo_disable_wrapper(f, {})
    assert wrapped(2) == 3
    assert getattr(wrapped, "_torchdynamo_disable", False)


def test_dynamo_eager_preserves_output_and_gradients(monkeypatch):
    compiler = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compiler(fn, backend="eager"))
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    def original(value, weight):
        return (value @ weight).sin()
    wrapped = compile_method_wrapper(original, {})
    x = torch.randn(3, 4, requires_grad=True)
    w = torch.randn(4, 5, requires_grad=True)
    expected = original(x, w)
    grads = torch.autograd.grad(expected.sum(), (x, w))
    actual = wrapped(x, w)
    actual_grads = torch.autograd.grad(actual.sum(), (x, w))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, grads):
        torch.testing.assert_close(actual_grad, expected_grad)
