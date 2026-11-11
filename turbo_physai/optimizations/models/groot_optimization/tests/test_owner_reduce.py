from types import SimpleNamespace
import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from groot_optimization.zero2_owner_reduce import (
    _fill_window, _store_window, is_balanced, owner_reduce_wrapper, reduce_by_owner,
)


def test_window_copy_and_padding():
    tensors = [torch.tensor([1., 2., 3.]), torch.tensor([4., 5.])]
    dest = torch.full((4,), -1.)
    _fill_window(tensors, 3, 4, dest)
    assert dest.tolist() == [4., 5., 0., 0.]
    _store_window(tensors, 2, 3, torch.tensor([7., 8., 9.]))
    assert tensors[0].tolist() == [1., 2., 7.]
    assert tensors[1].tolist() == [8., 9.]
    assert is_balanced([4, 4]) and not is_balanced([40, 1, 1, 1])


def test_empty_and_mismatched_owner_inputs():
    assert reduce_by_owner([], [], group=None, rank=0, world_size=2, dtype=torch.float32, staging={}) == "empty"
    with pytest.raises(ValueError):
        reduce_by_owner([torch.zeros(1)], [], group=None, rank=0, world_size=2, dtype=torch.float32, staging={})


def test_reduce_scatter_windows_convert_dtype_and_reuse_staging(monkeypatch):
    from groot_optimization import zero2_owner_reduce as module

    calls = []
    def reduce_scatter(output, packed, group=None):
        calls.append((packed.dtype, packed.numel()))
        output.copy_(packed.view(2, -1).sum(dim=0))

    monkeypatch.setattr(module.dist, "reduce_scatter_tensor", reduce_scatter)
    staging = {}
    tensors = [torch.arange(5, dtype=torch.float32), torch.arange(5, dtype=torch.float32) + 10]
    path = reduce_by_owner(tensors, [0, 1], group=None, rank=0, world_size=2,
                           dtype=torch.float64, staging=staging, max_elements=4)
    assert path == "reduce_scatter"
    torch.testing.assert_close(tensors[0], torch.arange(5, dtype=torch.float32) * 2 + 10)
    assert calls == [(torch.float64, 4), (torch.float64, 4), (torch.float64, 2)]
    pointers = {key: value.data_ptr() for key, value in staging.items()}
    reduce_by_owner([torch.ones(5), torch.ones(5)], [0, 1], group=None, rank=0, world_size=2,
                    dtype=torch.float64, staging=staging, max_elements=4)
    assert pointers == {key: value.data_ptr() for key, value in staging.items()}


@pytest.mark.parametrize("state,divide", [({}, False), ({"activated": True, "in_average": False}, False),
                                           ({"activated": True, "in_average": True}, True)])
def test_owner_wrapper_preserves_other_calls(state, divide):
    calls = []
    def original(self, bucket, communication_data_type, numel_per_bucket=1, log=None, divide=True, process_group=None):
        calls.append((bucket, divide))
        return "original"
    obj = SimpleNamespace(_turbo_physai_gro=state, dp_process_group="group")
    wrapped = owner_reduce_wrapper(original, {})
    assert wrapped(obj, [(0, torch.ones(1))], torch.float32, divide=divide, process_group="group") == "original"
    assert len(calls) == 1


def test_owner_wrapper_activates_and_keeps_optimizer_local_state(monkeypatch):
    from groot_optimization import zero2_owner_reduce as module
    fake_dist = SimpleNamespace(get_rank=lambda group: 0, get_world_size=lambda group: 2)
    monkeypatch.setattr(module, "dist", fake_dist)
    calls = []
    def fake_reduce(slices, owners, **kwargs):
        calls.append((slices, owners, kwargs["staging"]))
        return "reduce_scatter"
    monkeypatch.setattr(module, "reduce_by_owner", fake_reduce)
    def original(self, bucket, communication_data_type, numel_per_bucket=1, log=None, divide=True, process_group=None):
        pytest.fail("GRO bucket must use owner reduction")
    group = object()
    obj = SimpleNamespace(_turbo_physai_gro={"activated": True, "in_average": True}, dp_process_group=group)
    wrapped = owner_reduce_wrapper(original, {})
    wrapped(obj, [(0, torch.ones(1))], torch.float32, divide=False, process_group=group)
    wrapped(obj, [(1, torch.ones(1))], torch.float32, divide=False, process_group=group)
    assert obj._turbo_physai_obr["counts"]["reduce_scatter"] == 2
    assert calls[0][2] is calls[1][2] is obj._turbo_physai_obr["staging"]


def test_gro_average_flag_is_restored_on_error(monkeypatch):
    from contextlib import nullcontext
    from groot_optimization import zero2_overlap as module
    monkeypatch.setattr(module, "build_rank_and_offsets", lambda optimizer, bucket: ([(0, 0, 1)], 1))
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 1)
    class Tensor:
        def record_stream(self, stream): pass
        def numel(self): return 1
        def div_(self, value): return self
        def narrow(self, *args): return self
    stream = SimpleNamespace(wait_stream=lambda _: None)
    state = {"activated": True}
    optimizer = SimpleNamespace(reduction_stream=stream, ipg_buckets={"dtype": SimpleNamespace(has_moe_params=False)},
                                dp_process_group=None, sequence_parallel_size=1, reduce_bucket_size=1,
                                _turbo_physai_gro=state)
    def fail(*args, **kwargs):
        assert state["in_average"] is True
        raise RuntimeError("failed")
    optimizer.allreduce_and_scatter = fail
    impl = module._OverlapImplementation()
    impl.optimizer = optimizer
    impl.accelerator = SimpleNamespace(current_stream=lambda: stream, stream=lambda _: nullcontext())
    with pytest.raises(RuntimeError, match="failed"):
        impl.average_tensor(Tensor(), "dtype")
    assert state["in_average"] is False


def _gloo_worker(rank, world, init_path, lengths, expected_path, max_elements):
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=world)
    try:
        values = torch.arange(1, sum(lengths) + 1, dtype=torch.float32) * (rank + 1)
        slices, owners = [], []
        offset = 0
        for owner, length in enumerate(lengths):
            slices.append(values.narrow(0, offset, length))
            owners.append(owner)
            offset += length
        path = reduce_by_owner(slices, owners, staging={}, group=None, rank=rank,
                               world_size=world, dtype=torch.float32, max_elements=max_elements)
        assert path == expected_path
        offset = sum(lengths[:rank])
        expected = torch.arange(1, sum(lengths) + 1, dtype=torch.float32) * (world * (world + 1) // 2)
        torch.testing.assert_close(values[offset:offset + lengths[rank]], expected[offset:offset + lengths[rank]])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("lengths,path", [([4, 4], "reduce_scatter"), ([30, 1, 1, 1], "reduce")])
def test_gloo_owner_reduction(lengths, path):
    try:
        with socket.socket(socket.AF_INET) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except PermissionError:
        pytest.skip("sandbox does not permit Gloo socket setup")
    if not dist.is_gloo_available():
        pytest.skip("Gloo backend is unavailable")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_gloo_worker, args=(len(lengths), os.path.join(directory, "store"), lengths, path, 6),
                 nprocs=len(lengths), join=True)
