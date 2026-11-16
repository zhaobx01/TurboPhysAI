import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def benchmark():
    return importlib.import_module('benchmark')


def launcher():
    return importlib.import_module('rank_launcher')


def test_resource_wait_cli_runs_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / 'resource_wait.py'), '--help'],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert '--log' in result.stdout and '--devices' in result.stdout


def test_cases_resolve_catalog_dependencies_and_keep_baseline_strict():
    b = benchmark()
    assert len(b.CASES) == 22
    baseline = b.case_groups('baseline')
    assert baseline == {'groot.compat.training_config', 'groot.dit.hip_math_sdpa'}
    assert b.case_groups('pad_156') == baseline | {'groot.data.pad_156'}
    assert b.case_groups('obr') == baseline | {'groot.zero2.grad_reduce_overlap', 'groot.zero2.owner_reduce'}
    assert b.case_groups('text_forwards') == baseline | {
        'groot.compile.qwen_text_layer', 'groot.qwen.flash_attn_dynamo_disable', 'groot.qwen.disable_kv_cache'}
    assert 'groot.compat.opencv_video' not in b.case_groups('all_on')
    assert len(b.case_groups('all_on')) == 17


def test_measure_takes_slowest_rank_and_keeps_unstable_complete_runs():
    b = benchmark()
    records = [{'rank': i, 'world_size': 8, 'step_seconds': [.1 + .01*i] * 40} for i in range(8)]
    records[0]['step_seconds'][34] = 1.0
    result = b.measure(records)
    assert result['step35_ms'] == pytest.approx(1000)
    assert result['step35_samples_per_second'] == pytest.approx(960)
    assert result['mean31_40_ms'] == pytest.approx(253)
    assert result['mean31_40_samples_per_second'] == pytest.approx(960/.253)
    assert result['cv_pct'] > 5
    assert result['valid'] is True


def test_measure_rejects_incomplete_duplicate_and_nonfinite_records():
    b = benchmark()
    records = [{'rank': i, 'world_size': 8, 'step_seconds': [.1] * 40} for i in range(8)]
    for bad in [records[:-1], records[:-1] + [records[0]],
                records[:-1] + [dict(records[-1], step_seconds=[math.nan]*40)]]:
        with pytest.raises(ValueError):
            b.measure(bad)


def test_pending_attempts_skip_success_and_retry_failures_only_once():
    b = benchmark()
    rows = [{'case':'baseline','round':'speed-1','status':'success','attempt':1},
            {'case':'baseline','round':'speed-2','status':'failed','attempt':1},
            {'case':'baseline','round':'trace','status':'failed','attempt':1},
            {'case':'baseline','round':'trace','status':'failed','attempt':2}]
    assert b.next_attempt(rows, 'baseline', 'speed-1') is None
    assert b.next_attempt(rows, 'baseline', 'speed-2') == 2
    assert b.next_attempt(rows, 'baseline', 'trace') is None
    assert b.next_attempt(rows, 'baseline', 'trace', retry_failed=True) == 3


def test_aggregate_uses_successful_attempt_once_and_marks_single_round():
    b = benchmark()
    rows = [dict(case='baseline',round='speed-1',status='failed',attempt=1),
            dict(case='baseline',round='speed-1',status='success',attempt=2,step35_ms=10,
                 step35_samples_per_second=96,mean31_40_ms=20,mean31_40_samples_per_second=48),
            dict(case='baseline',round='speed-2',status='failed',attempt=1)]
    result = b.aggregate(rows, 'baseline')
    assert result['successful_rounds'] == 1
    assert result['step35_samples_per_second'] == 96
    assert result['label'] == '仅一轮'


def test_profiler_records_step_35_and_stops_while_training_continues(tmp_path):
    module = launcher()
    callback = module.TraceCallback(tmp_path, rank=0, activities=[torch.profiler.ProfilerActivity.CPU])
    callback.on_train_begin(None, None, None)
    try:
        for step in range(1, 41):
            with torch.profiler.record_function(f'step_{step}'):
                torch.ones(1).add_(step)
            callback.on_step_end(None, type('State', (), {'global_step': step})(), None)
        callback.on_train_end(None, None, None)
    finally:
        callback.close()
    trace = json.loads((tmp_path / 'trace_rank_0_iter_35.json').read_text())
    names = {event.get('name') for event in trace['traceEvents']}
    assert 'step_35' in names
    assert 'step_34' not in names
    assert 'step_36' not in names
    assert (tmp_path / 'rank0-key-averages.txt').read_text().strip()
    assert json.loads((tmp_path / 'profiler-stopped-rank-0.json').read_text()) == {
        'rank': 0, 'step': 35, 'stopped_after_trace': True}


def test_trace_validation_requires_every_rank_and_key_averages(tmp_path):
    b = benchmark()
    profile = tmp_path / 'profiling'
    profile.mkdir()
    for rank in range(8):
        (profile / f'trace_rank_{rank}_iter_35.json').write_text('{"traceEvents": []}')
        (profile / f'profiler-stopped-rank-{rank}.json').write_text(json.dumps({'rank': rank,'step':35,'stopped_after_trace':True}))
    (profile / 'rank0-key-averages.txt').write_text('CPU operator summary')
    assert b.validate_trace(tmp_path)['trace_step'] == 35
    (profile / 'trace_rank_7_iter_35.json').unlink()
    with pytest.raises(ValueError):
        b.validate_trace(tmp_path)


def test_resume_skips_success_and_rejects_changed_fingerprint(tmp_path, monkeypatch):
    b = benchmark()
    monkeypatch.setattr(b, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(b, 'fingerprint', lambda: {'model_sha':'expected'})
    monkeypatch.setattr(b, 'prepare_case', lambda output, case: tmp_path / 'config.yaml')
    calls = []
    def fake_attempt(output, case, round_name, number, config, expected):
        calls.append((round_name, number))
        path = output / case / round_name / f'attempt-{number:02d}'
        path.mkdir(parents=True)
        row = dict(case=case, round=round_name, attempt=number, status='success', directory=str(path),
                   step35_ms=10, step35_samples_per_second=96000, mean31_40_ms=20,
                   mean31_40_samples_per_second=48000, cv_pct=1, drift_pct=2,
                   max_memory_allocated_bytes=0, max_memory_reserved_bytes=0, device_peak_percent={}, trace_step=35)
        b.atomic_json(path / 'attempt.json', row)
        return row
    monkeypatch.setattr(b, 'run_attempt', fake_attempt)
    output = tmp_path / 'result'
    args = ['--output', str(output), '--cases', 'baseline']
    assert b.main(args) == 0
    assert calls == [('speed-1', 1), ('speed-2', 1), ('trace', 1)]
    calls.clear()
    assert b.main(args + ['--resume']) == 0
    assert calls == []
    monkeypatch.setattr(b, 'fingerprint', lambda: {'model_sha':'changed'})
    with pytest.raises(RuntimeError, match='fingerprint'):
        b.main(args + ['--resume'])


def test_resource_wait_query_failure_resets_idle_streak(tmp_path):
    import resource_wait
    card = {'vram Total Used Memory (MiB)': '1', 'vram Total Memory (MiB)': '100'}
    good = {f'card{i}': card for i in range(8)}
    observations = iter([good, good, RuntimeError('query failed'), good, good, good])
    def query():
        observation = next(observations)
        if isinstance(observation, Exception):
            raise observation
        return observation
    sleeps = []
    resource_wait.wait_for_idle_devices(tmp_path / 'wait.jsonl', query=query, sleep=sleeps.append)
    rows = [json.loads(x) for x in (tmp_path / 'wait.jsonl').read_text().splitlines()]
    assert [x['consecutive_idle'] for x in rows] == [1, 2, 0, 1, 2, 3]
    assert sleeps == [30] * 5


def test_process_group_termination():
    import os
    import subprocess
    import sys
    b = benchmark()
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    b.CHILD = process
    try:
        b.stop_child()
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
        b.CHILD = None


def test_launcher_disables_saves_and_propagates_training_failure(tmp_path, monkeypatch):
    import types
    module = launcher()
    from transformers.trainer_utils import SaveStrategy
    class FakeTrainer:
        def __init__(self):
            self.args = types.SimpleNamespace(save_strategy=SaveStrategy.STEPS)
            self.callbacks = []
        def add_callback(self, callback):
            self.callbacks.append(callback)
        def train(self):
            assert self.args.save_strategy == SaveStrategy.NO
            raise ValueError('original training error')
        def save_model(self):
            raise AssertionError('should be disabled')
    trainer_module = types.ModuleType('gr00t.experiment.trainer')
    trainer_module.Gr00tTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, 'gr00t.experiment.trainer', trainer_module)
    monkeypatch.setenv('GROOT_MODEL', '/model')
    monkeypatch.setenv('PHYS_BENCH_ATTEMPT_DIR', str(tmp_path))
    monkeypatch.setenv('PHYS_BENCH_TRACE', '0')
    def fake_entry(*args, **kwargs):
        trainer = FakeTrainer()
        assert trainer.save_model() is None
        with pytest.raises(ValueError, match='original training error'):
            trainer.train()
        assert len(trainer.callbacks) == 1
    monkeypatch.setattr(module.runpy, 'run_path', fake_entry)
    module.main()


def test_background_lock_rejects_duplicate_runner(tmp_path, monkeypatch):
    import fcntl
    b = benchmark()
    monkeypatch.setattr(b, 'DATA_ROOT', tmp_path)
    lock_dir = tmp_path / 'outputs'
    lock_dir.mkdir()
    with (lock_dir / '.physai-groot-benchmark.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already running'):
            b.main(['--output', str(tmp_path / 'out'), '--cases', 'baseline'])


def test_retry_uses_one_successful_attempt_and_keeps_failed_evidence(tmp_path, monkeypatch):
    b = benchmark()
    monkeypatch.setattr(b, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(b, 'fingerprint', lambda: {'model_sha':'expected'})
    monkeypatch.setattr(b, 'prepare_case', lambda output, case: tmp_path / 'config.yaml')
    calls = []
    def fake_attempt(output, case, round_name, number, config, expected):
        calls.append((round_name, number))
        status = 'failed' if round_name == 'speed-1' and number == 1 else 'success'
        path = output / case / round_name / f'attempt-{number:02d}'
        path.mkdir(parents=True)
        row = dict(case=case, round=round_name, attempt=number, status=status, directory=str(path),
                   step35_ms=10, step35_samples_per_second=96000, mean31_40_ms=20,
                   mean31_40_samples_per_second=48000, cv_pct=1, drift_pct=2,
                   max_memory_allocated_bytes=0, max_memory_reserved_bytes=0, device_peak_percent={}, trace_step=35)
        b.atomic_json(path / 'attempt.json', row)
        return row
    monkeypatch.setattr(b, 'run_attempt', fake_attempt)
    output = tmp_path / 'result'
    assert b.main(['--output', str(output), '--cases', 'baseline']) == 0
    assert calls == [('speed-1', 1), ('speed-1', 2), ('speed-2', 1), ('trace', 1)]
    data = json.loads((output / 'results.json').read_text())
    assert len(data['attempts']) == 4
    assert data['aggregates']['baseline']['successful_rounds'] == 2


def test_all_22_cases_match_requested_switches():
    b = benchmark()
    assert len(b.SINGLES) == 13
    for name, group in b.SINGLES.items():
        groups = b.case_groups(name)
        assert group in groups
        assert groups.intersection(set(b.SINGLES.values()) - {group}) <= {
            b.SINGLES['gro']}
    assert b.CASES['combo_runtime'] == {b.SINGLES[n] for n in ('pixel_bf16', 'fused_adam', 'gro')}
    assert b.CASES['combo_obr_dit_vl_text_allgather'] == b.CASES['combo_obr_dit_vl_text'] | {b.SINGLES['allgather']}
    assert b.case_groups('all_on') == set(b.SINGLES.values()) | b.BASELINE | {
        'groot.qwen.flash_attn_dynamo_disable', 'groot.qwen.disable_kv_cache'}


def test_relative_baseline_uses_successful_rounds_only(tmp_path):
    b = benchmark()
    def row(case, name, value):
        return dict(case=case, round=name, status='success', attempt=1, directory='x',
                    step35_ms=10, step35_samples_per_second=value, mean31_40_ms=10,
                    mean31_40_samples_per_second=value, cv_pct=50, drift_pct=50,
                    max_memory_allocated_bytes=0, max_memory_reserved_bytes=0, device_peak_percent={}, trace_step=35)
    rows = [row('baseline','speed-1',100), row('baseline','speed-2',200),
            row('pad_156','speed-1',300)]
    b.write_report(tmp_path, ['baseline','pad_156'], rows)
    data = json.loads((tmp_path / 'results.json').read_text())
    assert data['aggregates']['baseline']['mean31_40_samples_per_second'] == 150
    assert data['aggregates']['pad_156']['mean31_40_samples_per_second_improvement_pct'] == 100
    assert data['aggregates']['pad_156']['label'] == '仅一轮'


def test_case_generation_preserves_shared_recipe_and_saved_snapshot(tmp_path, monkeypatch):
    import subprocess
    import types
    import yaml
    b = benchmark()
    original = (b.ROOT / 'configs/recipe.yaml').read_bytes()
    def fake_run(command, **kwargs):
        if 'generate' in command:
            recipe = Path(command[command.index('--recipe') + 1])
            output = Path(command[command.index('--output') + 1])
            output.write_text(recipe.read_text())
            (output.parent / '.optimization.yaml.generation.json').write_text('{}')
        return types.SimpleNamespace(returncode=0, stdout='ok')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    config = b.prepare_case(tmp_path, 'obr')
    assert (b.ROOT / 'configs/recipe.yaml').read_bytes() == original
    data = yaml.safe_load(config.read_text())
    assert {g['id'] for g in data['optimization_groups'] if g['enabled']} == b.case_groups('obr')
    recipe = config.parent / 'recipe.yaml'
    first = recipe.read_bytes()
    assert b.prepare_case(tmp_path, 'obr') == config
    assert recipe.read_bytes() == first
    recipe.write_text(recipe.read_text().replace('enabled: true', 'enabled: false', 1))
    with pytest.raises(RuntimeError, match='recipe changed'):
        b.prepare_case(tmp_path, 'obr')


def test_snapshot_integrity_detects_runtime_and_generated_config_edits(tmp_path):
    b = benchmark()
    output = tmp_path / 'output'
    folder = output / 'case' / 'configs'
    folder.mkdir(parents=True)
    (output / 'runtime.yaml').write_bytes((b.ROOT / 'configs/runtime.yaml').read_bytes())
    for name in ('recipe.yaml', 'optimization.yaml', '.optimization.yaml.generation.json'):
        (folder / name).write_text(name)
    config = folder / 'optimization.yaml'
    (folder / 'snapshot.sha256').write_text(b.case_snapshot_digest(config))
    b.verify_snapshots(output, config)
    config.write_text('changed')
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        b.verify_snapshots(output, config)
    config.write_text('optimization.yaml')
    (output / 'runtime.yaml').write_text('changed')
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        b.verify_snapshots(output, config)


def test_report_uses_separate_reference_style_tables_without_excluding_unstable_runs(tmp_path):
    b = benchmark()
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    (attempt / 'device-memory.jsonl').write_text(json.dumps({'cards': {
        'card0': {'vram Total Used Memory (MiB)': '1200'}}}) + '\n' + json.dumps({'error':'unavailable'}) + '\n')
    row = dict(case='baseline', round='speed-1', status='success', attempt=1,
               directory=str(attempt), step35_ms=1000, step35_samples_per_second=960,
               mean31_40_ms=1200, mean31_40_samples_per_second=800, cv_pct=10,
               drift_pct=12, device_peak_percent={}, max_memory_allocated_bytes=1048576,
               max_memory_reserved_bytes=2097152)
    trace = dict(case='baseline', round='trace', status='success', attempt=1,
                 directory=str(attempt), trace_step=35)
    b.write_report(tmp_path, ['baseline', 'pad_156'], [row, trace])
    report = (tmp_path / 'report.md').read_text()
    for title in ('## 第 35 步', '## 第 31–40 步均值', '## 每轮明细', '## 测速显存', '## Trace', '## 方案 Group'):
        assert title in report
    assert '仅一轮' in report
    assert '1200.000' in report
    assert '1.000' in report and '2.000' in report
    assert '1/1' in report
    assert '10.000' in report and '12.000' in report
    assert '不按 CV 或漂移剔除' in report
    assert 'baseline / trace' in report


def test_background_entry_sets_four_samples_and_35_seconds(tmp_path):
    import os
    import subprocess
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    python = fake_bin / 'python3'
    python.write_text('#!/bin/sh\nprintf "%s %s\\n" "$GR00T_BENCH_IDLE_INTERVAL" "$GR00T_BENCH_IDLE_SAMPLES"\n')
    python.chmod(0o755)
    result = subprocess.run(['bash', str(SCRIPTS / 'run_benchmark_b960.sh'), '--worker', str(tmp_path / 'worker'), '--dry-run'],
                            env={**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH']},
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '35 4'
