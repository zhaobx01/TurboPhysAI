from pathlib import Path

from turbo_physai.runtime import load_runtime_config, prepare_environment


def test_groot_runtime_prepares_offline_environment_and_allows_overrides(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    runtime = load_runtime_config(root / "configs" / "runtime.yaml")
    monkeypatch.setenv("VIDEO_DECODER_BACKEND", "opencv")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/old-cache")
    env = prepare_environment(runtime, overrides={"VLM_MODEL_PATH": "/new-model"})
    assert env["VLM_MODEL_PATH"] == "/new-model"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["USE_WANDB"] == "0"
    assert "VIDEO_DECODER_BACKEND" not in env
    assert "TORCHINDUCTOR_CACHE_DIR" not in env
    assert "TURBO_PHYSAI_NUMA_AUTO" not in env
