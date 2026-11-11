import importlib.util

import pytest

from groot_optimization import catalog
from turbo_physai.engine.execution.replacements import default_handlers


@pytest.mark.model_deps
@pytest.mark.parametrize("declaration", [
    catalog.DIT_HIP_MATH_SDPA, catalog.QWEN_FAST_POS, catalog.QWEN_ROT_POS,
    catalog.QWEN_ROPE, catalog.FLASH_DISABLE, catalog.COMPILE_VISION,
    catalog.COMPILE_DIT, catalog.COMPILE_VL, catalog.PIXEL_VALUES_BF16, catalog.FIXED_PADDING,
    catalog.APEX_FUSED_ADAM, catalog.ZERO2_OVERLAP, catalog.DISABLE_KV_CACHE,
    catalog.COMPILE_TEXT, catalog.ZERO2_OWNER_REDUCE, catalog.ZERO2_ALLGATHER,
], ids=lambda item: item.group_id)
def test_handler_applies_and_restores_all_targets(declaration):
    for module in ("gr00t", "transformers", "deepspeed"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed")
    handlers = default_handlers()
    prepared = []
    try:
        for spec in declaration.specs:
            handler = handlers[spec.mechanism]
            item = handler.prepare(spec, {})
            snapshot = handler.snapshot(item)
            prepared.append((handler, snapshot))
            handler.apply(item)
            for target in item.targets:
                value = target.parent[target.attribute] if target.is_mapping else getattr(target.parent, target.attribute)
                assert value is item.replacement
    finally:
        for handler, snapshot in reversed(prepared):
            handler.restore(snapshot)
            for target in snapshot:
                value = target.parent[target.attribute] if target.is_mapping else getattr(target.parent, target.attribute)
                assert value is target.original


@pytest.mark.model_deps
def test_gro_and_owner_apply_together_and_restore_in_reverse():
    pytest.importorskip("deepspeed")
    handlers = default_handlers()
    prepared = []
    try:
        for declaration in (catalog.ZERO2_OVERLAP, catalog.ZERO2_OWNER_REDUCE):
            for spec in declaration.specs:
                handler = handlers[spec.mechanism]
                item = handler.prepare(spec, {})
                snapshot = handler.snapshot(item)
                prepared.append((handler, snapshot, item))
                handler.apply(item)
        for _, _, item in prepared:
            for target in item.targets:
                assert getattr(target.parent, target.attribute) is item.replacement
    finally:
        for handler, snapshot, _ in reversed(prepared):
            handler.restore(snapshot)
            for target in snapshot:
                assert getattr(target.parent, target.attribute) is target.original
