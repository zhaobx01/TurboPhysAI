from types import SimpleNamespace
import pytest
from groot_optimization.zero2_overlap import BucketEventScheduler, _runtime_state


class Event:
    def record(self, stream): self.stream = stream


class Stream:
    def __init__(self): self.waited = []
    def wait_event(self, event): self.waited.append(event)


def test_double_buffer_waits_only_before_reuse():
    stream = Stream()
    scheduler = BucketEventScheduler(lambda: stream, Event)
    scheduler.record_consumed(("bf16", 0), object())
    assert not scheduler.wait_before_reuse(("bf16", 1))
    assert scheduler.wait_before_reuse(("bf16", 0))
    assert len(stream.waited) == 1


def test_unsupported_optimizer_fails_or_falls_back():
    obj = SimpleNamespace()
    with pytest.raises(RuntimeError, match="Unsupported"):
        _runtime_state(obj, strict=True)
    other = SimpleNamespace()
    assert _runtime_state(other, strict=False) is None
    assert other._turbo_physai_gro["activated"] is False


def test_wrappers_refuse_uninitialized_optimizer():
    from groot_optimization.zero2_overlap import reduce_wrapper
    def original(self, param, i):
        return "original"
    with pytest.raises(RuntimeError, match="not initialized"):
        reduce_wrapper(original, {})(SimpleNamespace(), object(), 0)


def test_layout_difference_across_ranks_fails_before_activation(monkeypatch):
    from groot_optimization import zero2_overlap as z
    monkeypatch.setattr(z, "_unsupported_reasons", lambda obj: ["unsupported"])
    monkeypatch.setattr(z.dist, "is_available", lambda: True)
    monkeypatch.setattr(z.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(z.dist, "get_world_size", lambda: 2)
    def gather(result, status):
        result[:] = [status, {**status, "ok": True, "layout": "different"}]
    monkeypatch.setattr(z.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank applicability differs"):
        z._runtime_state(SimpleNamespace(), strict=False)


def test_average_tensor_scales_and_dispatches_owner_slices(monkeypatch):
    from contextlib import nullcontext
    from groot_optimization import zero2_overlap as z

    class FakeTensor:
        def __init__(self, values):
            self.values = values
            self.stream = None
        def numel(self): return len(self.values)
        def div_(self, n):
            self.values[:] = [x / n for x in self.values]
            return self
        def narrow(self, dim, start, size): return self.values[start:start + size]
        def record_stream(self, stream): self.stream = stream
    class ReductionStream:
        def wait_stream(self, stream): self.waited = stream
    parameter = SimpleNamespace(numel=lambda: 4)
    bucket = SimpleNamespace(has_moe_params=False, params=[(0, 0, 42)])
    calls = []
    reduction = ReductionStream()
    optimizer = SimpleNamespace(
        reduction_stream=reduction, ipg_buckets={"bf16": bucket},
        bit16_groups=[[parameter]], param_to_partition_ids=[{42: [0, 1]}],
        grad_start_offset=[{0: {42: 0}, 1: {42: 2}}],
        dp_process_group="group", sequence_parallel_size=1, reduce_bucket_size=8,
        allreduce_and_scatter=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    accelerator = SimpleNamespace(current_stream=lambda: "compute", stream=lambda stream: nullcontext())
    monkeypatch.setattr(z.dist, "get_world_size", lambda group=None: 2)
    impl = z._OverlapImplementation()
    impl.optimizer, impl.accelerator = optimizer, accelerator
    tensor = FakeTensor([2, 4, 6, 8])
    impl.average_tensor(tensor, "bf16")
    assert tensor.values == [1, 2, 3, 4]
    assert tensor.stream is reduction and reduction.waited == "compute"
    assert calls[0][0][0] == [(0, [1, 2]), (1, [3, 4])]
    assert calls[0][1]["divide"] is False
