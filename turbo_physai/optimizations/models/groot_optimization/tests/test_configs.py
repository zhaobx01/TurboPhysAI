from pathlib import Path

import pytest
import yaml

from turbo_physai.engine.config.generation_record import verify_generated


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_config_receipt_is_valid():
    assert verify_generated(CONFIGS / "optimization.yaml")["model_commit"] == "d9bdcf9686649761440496445dd60c1bbb225ea1"


def test_generated_config_matches_recipe_group_enablement():
    recipe = yaml.safe_load((CONFIGS / "recipe.yaml").read_text())
    config = yaml.safe_load((CONFIGS / "optimization.yaml").read_text())
    recipe_groups = {g["id"]: g["enabled"] for g in recipe["optimization_groups"]}
    config_groups = {g["id"]: g["enabled"] for g in config["optimization_groups"]}
    assert len(config_groups) == 16
    assert config_groups == recipe_groups


@pytest.mark.model_deps
def test_all_groups_prepare_without_conflicts(tmp_path):
    """Enable every Group in a temporary copy of the recipe; no second config is shipped."""
    import importlib.util
    for name in ("gr00t", "transformers", "deepspeed", "apex", "amp_C"):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"{name} is unavailable")
    from groot_optimization import catalog  # noqa: F401
    from turbo_physai.engine.checking.context import detect_context
    from turbo_physai.engine.checking.ordering import Preparation
    from turbo_physai.engine.config.loader import load_optimization_config
    from turbo_physai.engine.contracts import Decision
    from turbo_physai.engine.definitions.registry import default_registry
    from turbo_physai.engine.execution.replacements import default_handlers

    recipe = yaml.safe_load((CONFIGS / "recipe.yaml").read_text())
    for group in recipe["optimization_groups"]:
        group["enabled"] = True
    path = tmp_path / "all_enabled.yaml"
    path.write_text(yaml.safe_dump(recipe))

    prepared = Preparation(default_registry, default_handlers()).prepare(
        run_id="groot-integration-check",
        config=load_optimization_config(path),
        environment=detect_context(),
    )
    assert not prepared.conflicts
    assert len(prepared.groups) == 16
    assert all(group.decision is Decision.APPLY for group in prepared.groups)
