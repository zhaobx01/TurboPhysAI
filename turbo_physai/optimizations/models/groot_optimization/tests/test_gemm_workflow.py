import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_gemm_workflow_b960.sh'


def workflow(monkeypatch):
    source = SCRIPT.read_text().split("<<'GEMM_WORKFLOW_PY'\n", 1)[1].rsplit('\nGEMM_WORKFLOW_PY', 1)[0]
    import importlib
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    benchmark = importlib.import_module('benchmark')
    monkeypatch.setattr(benchmark, 'case_groups', benchmark.case_groups)
    module = types.ModuleType('gemm_workflow_test')
    monkeypatch.setenv('PHYS_GEMM_SCRIPT', str(SCRIPT))
    exec(compile(source, str(SCRIPT), 'exec'), module.__dict__)
    return module


def test_plan_is_padded_and_contains_exactly_two_rounds():
    result = subprocess.run(['bash', str(SCRIPT), 'plan', '--cases', 'obr', 'pixel_bf16', 'obr'],
                            text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['cases'] == ['baseline', 'obr', 'pixel_bf16']
    assert plan['rounds'] == ['speed-1', 'speed-2']
    assert plan['capture_case'] == 'combo_obr_dit_vl_text_allgather'
    assert all('groot.data.pad_156' in groups for groups in plan['groups'].values())
    assert 'groot.zero2.grad_reduce_overlap' in plan['groups']['obr']
    assert plan['trace'] is False


def test_clean_layer3_preserves_distinct_calls_and_rejects_invalid(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    raw = tmp_path / 'raw.log'
    cleaned = tmp_path / 'clean.log'
    command = './rocblas-bench -f gemm_ex -m 1 -n 2 -k 3'
    raw.write_text(f'prefix {command}\n{command} {command}\n')
    assert g.clean_layer3(raw, cleaned) == 1
    assert cleaned.read_text().strip() == command
    raw.write_text('./rocblas-bench -f gemm_ex -m 1\n')
    with pytest.raises(ValueError, match='GEMM'):
        g.clean_layer3(raw, cleaned)


def test_retry_stops_after_two_and_interrupt_never_retries(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    seen = []
    def fail(number, directory):
        seen.append(number)
        raise ValueError('bad')
    with pytest.raises(RuntimeError, match='bad'):
        g.retry(tmp_path, fail)
    assert seen == [1, 2]
    seen.clear()
    def interrupt(number, directory):
        seen.append(number)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        g.retry(tmp_path / 'interrupted', interrupt)
    assert seen == [1]


def test_resume_identity_and_selected_config_integrity(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'G01.dat').write_text('initial')
    digest = g.tree_digest(config)
    g.verify_config(config, digest)
    (config / 'G01.dat').write_text('changed')
    with pytest.raises(RuntimeError, match='fingerprint'):
        g.verify_config(config, digest)
    with pytest.raises(RuntimeError, match='fingerprint'):
        g.check_manifest({'fingerprint': 'a', 'cases': ['baseline']},
                         {'fingerprint': 'b', 'cases': ['baseline']})


def test_tuning_command_never_updates_installed_library(monkeypatch):
    g = workflow(monkeypatch)
    command = g.tuning_command(Path('/input.log'))
    assert '--input' in command and '/input.log' in command
    assert 'update_target_config=0' in command
    assert 'update_target_config=1' not in command


def test_simulated_workflow_runs_ordered_two_rounds_and_resumes(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    monkeypatch.setattr(g.bench, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(g, 'identity', lambda cases: {'cases': cases, 'fingerprint': 'fixed', 'installed_library_sha256': 'installed'})
    monkeypatch.setattr(g, 'installed_unchanged', lambda manifest: None)
    monkeypatch.setattr(g, 'snapshot_shared', lambda path: path.mkdir(parents=True, exist_ok=True))
    calls = []
    def capture(number, directory):
        calls.append(('capture', number))
        input_path = directory / 'layer3.clean.log'
        input_path.write_text('./rocblas-bench -f gemm_ex -m 1 -n 2 -k 3\n')
        return {'input': str(input_path), 'input_sha256': g.sha(input_path)}
    def tune(number, directory, capture, manifest, root):
        calls.append(('tune', number))
        target = directory / 'selected_config'
        target.mkdir()
        (target / 'G01.dat').write_text('optimized')
        return {'config': str(target), 'config_sha256': g.tree_digest(target), 'input_sha256': capture['input_sha256']}
    def prepare(output, case):
        return output / case / 'configs/optimization.yaml'
    def train(output, case, round_name, number, config, fingerprint):
        calls.append((case, round_name, number))
        assert os.environ['ROCBLAS_TENSILE_LIBPATH'].endswith('/selected_config')
        assert 'ROCBLAS_LAYER' not in os.environ
        assert 'groot.data.pad_156' in g.bench.case_groups(case)
        directory = output / case / round_name / f'attempt-{number:02d}'
        directory.mkdir(parents=True)
        row = dict(case=case, round=round_name, attempt=number, status='success', directory=str(directory),
                   step35_ms=1000, step35_samples_per_second=960, mean31_40_ms=1200,
                   mean31_40_samples_per_second=800, cv_pct=9, drift_pct=10,
                   max_memory_allocated_bytes=0, max_memory_reserved_bytes=0)
        g.save(directory / 'attempt.json', row)
        return row
    monkeypatch.setattr(g, 'capture_attempt', capture)
    monkeypatch.setattr(g, 'tune_attempt', tune)
    monkeypatch.setattr(g.bench, 'prepare_case', prepare)
    monkeypatch.setattr(g.bench, 'run_attempt', train)
    monkeypatch.setattr(g.bench, 'fingerprint', lambda: {'model_sha': g.bench.SHA})
    output = tmp_path / 'result'
    cases = g.selected_cases(['obr', 'pixel_bf16'])
    g.workflow(output, cases, False)
    assert calls == [('capture', 1), ('tune', 1), ('baseline', 'speed-1', 1), ('baseline', 'speed-2', 1),
                     ('obr', 'speed-1', 1), ('obr', 'speed-2', 1),
                     ('pixel_bf16', 'speed-1', 1), ('pixel_bf16', 'speed-2', 1)]
    report = (output / 'benchmark/report.md').read_text()
    assert '兼容项＋pad_156' in report and '本次不采集 Trace' in report
    assert read_json(output / 'benchmark/results.json')['aggregates']['baseline']['successful_rounds'] == 2
    calls.clear()
    g.workflow(output, cases, True)
    assert calls == []


def read_json(path):
    return json.loads(path.read_text())


def test_tuning_simulation_archives_config_and_preserves_installed_library(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    tool = tmp_path / 'tool'
    tool.mkdir()
    lib = tmp_path / 'library_gpu5'
    lib.mkdir()
    (lib / 'G-original.dat').write_text('original')
    monkeypatch.setattr(g, 'TOOL', tool)
    monkeypatch.setattr(g, 'LIBRARY', lib)
    waits = []
    monkeypatch.setattr(g.bench.resource_wait, 'wait_for_idle_devices', lambda path, **kw: waits.append(path))
    def run(command, env, log_path):
        generated = tool / 'optimization_configs/new/config/library_gpu5'
        generated.mkdir(parents=True)
        (generated / 'G-new.dat').write_text('optimized')
        log_path.write_text('所有流程执行完毕')
    monkeypatch.setattr(g, 'run_process', run)
    input_path = tmp_path / 'layer3.log'
    input_path.write_text('data')
    capture = {'input': str(input_path), 'input_sha256': g.sha(input_path)}
    manifest = {'installed_library_sha256': g.tree_digest(lib), 'cases': ['baseline']}
    monkeypatch.setattr(g, 'identity', lambda cases: manifest)
    attempt = tmp_path / 'attempt-01'
    attempt.mkdir()
    result = g.tune_attempt(1, attempt, capture, manifest, tmp_path)
    assert len(waits) == 1
    assert (Path(result['config']) / 'G-new.dat').read_text() == 'optimized'
    assert (lib / 'G-original.dat').read_text() == 'original'
    assert result['config_sha256'] == g.tree_digest(Path(result['config']))


def test_locks_reject_other_gemm_workflow(tmp_path, monkeypatch):
    import fcntl
    g = workflow(monkeypatch)
    monkeypatch.setattr(g.bench, 'DATA_ROOT', tmp_path)
    folder = tmp_path / 'outputs'
    folder.mkdir()
    with (folder / '.gemm-workflow.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='锁'):
            with g.workflow_locks():
                pass


def test_capture_uses_padded_combo_and_layer3_only_for_collection(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    monkeypatch.setattr(g.bench, 'prepare_case', lambda work, case: work / 'config.yaml')
    monkeypatch.setattr(g.bench, 'fingerprint', lambda: {})
    def fake_run(work, case, round_name, number, config, fingerprint):
        assert case == g.CAPTURE and round_name == 'speed-1'
        assert os.environ['ROCBLAS_LAYER'] == '3'
        assert 'ROCBLAS_TENSILE_LIBPATH' not in os.environ
        assert g.PAD in g.bench.case_groups(case)
        train = work / 'fake'
        train.mkdir()
        (train / 'train.log').write_text('./rocblas-bench -f gemm_ex -m 1 -n 2 -k 3\n')
        return {'status': 'success', 'directory': str(train)}
    monkeypatch.setattr(g.bench, 'run_attempt', fake_run)
    monkeypatch.setenv('ROCBLAS_TENSILE_LIBPATH', '/old')
    result = g.capture_attempt(1, tmp_path)
    assert result['gemm_commands'] == 1
    assert os.environ['ROCBLAS_TENSILE_LIBPATH'] == '/old'


def test_failed_tuning_is_backed_up_on_retry(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    tool = tmp_path / 'tool'
    tool.mkdir()
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'G-original.dat').write_text('original')
    monkeypatch.setattr(g, 'TOOL', tool)
    monkeypatch.setattr(g, 'LIBRARY', lib)
    monkeypatch.setattr(g.bench.resource_wait, 'wait_for_idle_devices', lambda *a, **kw: None)
    calls = []
    def run(command, env, log):
        calls.append(1)
        directory = tool / 'optimization_artifacts'
        directory.mkdir(exist_ok=True)
        (directory / 'evidence').write_text('run-' + str(len(calls)))
        if len(calls) == 1:
            raise RuntimeError('tune failed')
        generated = tool / 'optimization_configs/new/config/library_gpu5'
        generated.mkdir(parents=True)
        (generated / 'G.dat').write_text('tuned')
        log.write_text('所有流程执行完毕')
    monkeypatch.setattr(g, 'run_process', run)
    path = tmp_path / 'input'
    path.write_text('gemm')
    capture = {'input': str(path), 'input_sha256': g.sha(path)}
    manifest = {'installed_library_sha256': g.tree_digest(lib), 'cases': ['baseline']}
    monkeypatch.setattr(g, 'identity', lambda cases: manifest)
    result = g.retry(tmp_path / 'tuning', lambda n, d: g.tune_attempt(n, d, capture, manifest, tmp_path))
    assert len(calls) == 2
    assert (tmp_path / 'tuning/attempt-02/before_retry/optimization_artifacts/evidence').read_text() == 'run-1'
    assert Path(result['config']).is_dir()


def test_speed_attempt_retries_and_only_success_counts(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    selected_dir = tmp_path / 'selected'
    selected_dir.mkdir()
    (selected_dir / 'G.dat').write_text('config')
    selected = {'config': str(selected_dir), 'config_sha256': g.tree_digest(selected_dir)}
    manifest = {'fingerprint': 'same'}
    monkeypatch.setattr(g, 'identity', lambda cases: manifest)
    monkeypatch.setattr(g.bench, 'fingerprint', lambda: {})
    calls = []
    def run(output, case, round_name, number, config, fingerprint):
        calls.append((number, os.environ['ROCBLAS_TENSILE_LIBPATH']))
        directory = output / case / round_name / f'attempt-{number:02d}'
        directory.mkdir(parents=True)
        status = 'failed' if number == 1 else 'success'
        row = dict(case=case, round=round_name, attempt=number, status=status, directory=str(directory))
        if status == 'success':
            row.update(step35_ms=1000, step35_samples_per_second=960, mean31_40_ms=1200,
                       mean31_40_samples_per_second=800, cv_pct=8, drift_pct=7,
                       max_memory_allocated_bytes=0, max_memory_reserved_bytes=0)
        else:
            row['reason'] = 'synthetic failure'
        return row
    monkeypatch.setattr(g.bench, 'run_attempt', run)
    rows = []
    cases = ['baseline']
    result = g.retry(tmp_path / 'attempts', lambda n, d: g.speed_attempt(
        n, d, tmp_path, 'baseline', 'speed-1', tmp_path / 'config', selected, rows, cases, manifest))
    assert [n for n, _ in calls] == [1, 2]
    assert all(path == str(selected_dir) for _, path in calls)
    assert result['training']['status'] == 'success'
    data = read_json(tmp_path / 'benchmark/results.json')
    assert len(data['attempts']) == 2
    assert data['aggregates']['baseline']['successful_rounds'] == 1
    assert data['aggregates']['baseline']['label'] == '仅一轮'


def test_changed_fingerprint_stops_without_retry(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    seen = []
    def changed(number, directory):
        seen.append(number)
        raise RuntimeError('configuration fingerprint changed')
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        g.retry(tmp_path, changed)
    assert seen == [1]


def test_terminate_process_group_stops_child(tmp_path, monkeypatch):
    import signal
    import time
    g = workflow(monkeypatch)
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        g.terminate_group(proc)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def test_resume_ignores_uncommitted_training_result(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    folder = tmp_path / 'attempts/baseline/speed-1/attempt-01'
    folder.mkdir(parents=True)
    row = {'case': 'baseline', 'round': 'speed-1', 'attempt': 1, 'status': 'success', 'directory': 'x'}
    g.save(folder / 'training-result.json', row)
    assert g.load_speed_rows(tmp_path) == []
    g.save(folder / 'attempt.json', {'attempt': 1, 'status': 'failed', 'reason': 'validation failed'})
    restored = g.load_speed_rows(tmp_path)
    assert len(restored) == 1 and restored[0]['status'] == 'failed'
    assert restored[0]['reason'] == 'validation failed'


def test_tuning_retry_refuses_changed_shared_intermediates(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    tool = tmp_path / 'tool'
    tool.mkdir()
    artifact = tool / 'optimization_artifacts'
    artifact.mkdir()
    (artifact / 'item').write_text('before')
    monkeypatch.setattr(g, 'TOOL', tool)
    previous = tmp_path / 'attempt-01'
    previous.mkdir()
    (previous / 'launch.json').write_text('{}')
    (previous / 'shared-state.sha256').write_text(g.shared_digest())
    (artifact / 'item').write_text('external-change')
    attempt = tmp_path / 'attempt-02'
    attempt.mkdir()
    input_path = tmp_path / 'input'
    input_path.write_text('gemm')
    monkeypatch.setattr(g, 'installed_unchanged', lambda manifest: None)
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        g.tune_attempt(2, attempt, {'input': str(input_path), 'input_sha256': g.sha(input_path)}, {}, tmp_path)


def test_speed_rejects_source_change_before_accepting_result(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    selected_dir = tmp_path / 'selected'
    selected_dir.mkdir()
    (selected_dir / 'G.dat').write_text('config')
    selected = {'config': str(selected_dir), 'config_sha256': g.tree_digest(selected_dir)}
    manifest = {'fingerprint': 'before'}
    identities = iter([manifest, {'fingerprint': 'changed'}])
    monkeypatch.setattr(g, 'identity', lambda cases: next(identities))
    monkeypatch.setattr(g.bench, 'fingerprint', lambda: {})
    def run(output, case, round_name, number, config, fingerprint):
        directory = output / case / round_name / f'attempt-{number:02d}'
        directory.mkdir(parents=True)
        return {'case': case, 'round': round_name, 'attempt': number, 'status': 'success', 'directory': str(directory)}
    monkeypatch.setattr(g.bench, 'run_attempt', run)
    attempt = tmp_path / 'outer'
    attempt.mkdir()
    rows = []
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        g.speed_attempt(1, attempt, tmp_path, 'baseline', 'speed-1', tmp_path / 'config', selected, rows, ['baseline'], manifest)
    assert rows == []
    assert not (attempt / 'training-result.json').exists()


def test_installed_library_change_is_detected(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'G.dat').write_text('original')
    monkeypatch.setattr(g, 'LIBRARY', lib)
    manifest = {'installed_library_sha256': g.tree_digest(lib)}
    g.installed_unchanged(manifest)
    (lib / 'G.dat').write_text('changed')
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        g.installed_unchanged(manifest)


def test_failed_capture_stops_and_keeps_status_and_pending_report(tmp_path, monkeypatch):
    g = workflow(monkeypatch)
    monkeypatch.setattr(g.bench, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(g, 'identity', lambda cases: {'cases': cases, 'fingerprint': 'fixed'})
    calls = []
    def fail(number, directory):
        calls.append(number)
        raise RuntimeError('capture failed')
    monkeypatch.setattr(g, 'capture_attempt', fail)
    output = tmp_path / 'result'
    with pytest.raises(RuntimeError, match='capture failed'):
        g.workflow(output, ['baseline'], False)
    assert calls == [1, 2]
    assert read_json(output / 'state.json')['status'] == 'failed'
    assert (output / 'exit.code').read_text().strip() == '1'
    assert '待运行' in (output / 'benchmark/report.md').read_text()
    assert len(list((output / 'capture').glob('attempt-*/attempt.json'))) == 2
