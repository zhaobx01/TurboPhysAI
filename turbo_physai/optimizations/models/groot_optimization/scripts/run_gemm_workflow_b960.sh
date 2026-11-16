#!/usr/bin/env bash
# 独立编排：PhysAI GEMM 采集 → DTK 选优 → 各指定方案两轮测速；不采 Trace。
set -euo pipefail
export PHYS_GEMM_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
exec "${PHYS_GEMM_PYTHON:-$(command -v python3)}" -u - "$@" <<'GEMM_WORKFLOW_PY'
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

SCRIPT = Path(os.environ['PHYS_GEMM_SCRIPT']).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import benchmark as bench

CAPTURE = 'combo_obr_dit_vl_text_allgather'
PAD = 'groot.data.pad_156'
DTK = Path(os.environ.get('DTKROOT', '/opt/dtk')).resolve()
TOOL = DTK / 'lib/rocblas/auto_select_test/auto_select_tools'
AUTOTUNE = Path(os.environ.get('PHYS_GEMM_AUTOTUNE', '/workspace/Isaac-GR00T/getting_started/optimization/dcu_gemm_autotune.sh')).resolve()
LIBRARY = DTK / 'lib/rocblas/library_gpu5'
TUNING = {'strategy': 'common', 'chunk_size': '10', 'tune_warmup_times': '2',
          'tune_loop_times': '4', 'bench_warmup_times': '2', 'bench_loop_times': '10',
          'test_times': '2', 'enable_bestkernel_csv': '0', 'enable_mk_n_threshold': '0',
          'update_target_config': '0'}
SHARED = ('optimization_configs', 'optimization_artifacts', 'optimization_results', 'rocblas_layer3_size', 'configs')
_original_groups = bench.case_groups
bench.case_groups = lambda case: _original_groups(case) | {PAD}
CHILD = None


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bench.atomic_json(path, data)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path):
    path = Path(path)
    if not path.is_dir():
        raise RuntimeError(f'缺少目录: {path}')
    files = [(p.relative_to(path).as_posix(), sha(p)) for p in sorted(path.rglob('*')) if p.is_file()]
    if not files:
        raise RuntimeError(f'目录为空: {path}')
    return hashlib.sha256(json.dumps(files, ensure_ascii=False).encode()).hexdigest()


def verify_config(path, expected):
    if tree_digest(path) != expected:
        raise RuntimeError(f'Configuration fingerprint changed: {path}')


def clean_layer3(source, dest):
    commands = set()
    with Path(source).open(errors='replace') as stream:
        for line in stream:
            matches = list(re.finditer(r'\./rocblas-bench(?=\s)', line))
            for i, match in enumerate(matches):
                command = line[match.start():matches[i + 1].start() if i + 1 < len(matches) else len(line)].strip()
                try:
                    args = shlex.split(command)
                except ValueError as exc:
                    raise ValueError(f'无效 GEMM 命令: {command[:200]}') from exc
                if not all(flag in args for flag in ('-f', '-m', '-n', '-k')):
                    raise ValueError(f'GEMM 命令缺少参数: {command[:200]}')
                commands.add(command)
    if not commands:
        raise ValueError('Layer3 日志没有有效 GEMM 命令')
    Path(dest).write_text('\n'.join(sorted(commands)) + '\n')
    return len(commands)


def selected_cases(cases):
    if not cases:
        raise ValueError('必须使用 --cases 指定至少一个方案')
    invalid = set(cases) - set(bench.CASES)
    if invalid:
        raise ValueError(f'未知方案: {sorted(invalid)}')
    return ['baseline'] + [case for case in dict.fromkeys(cases) if case != 'baseline']


def plan(cases):
    cases = selected_cases(cases)
    return {'model_sha': bench.SHA, 'capture_case': CAPTURE, 'capture_steps': 40,
            'capture_groups': sorted(bench.case_groups(CAPTURE)), 'cases': cases,
            'groups': {case: sorted(bench.case_groups(case)) for case in cases},
            'gpus': 8, 'global_batch_size': 960, 'rounds': ['speed-1', 'speed-2'],
            'trace': False, 'tuning': TUNING,
            'idle': '8 卡显存各低于 5%，每 30 秒检查，连续 3 次',
            'baseline': '兼容项＋pad_156', 'autotune': str(AUTOTUNE)}


def tuning_command(input_path):
    cmd = ['bash', str(AUTOTUNE), 'run', '--source', 'layer3', '--hardware', 'BW',
           '--gpus', '0 1 2 3 4 5 6 7', '--cu-count', '0', '--input', str(input_path)]
    for key, value in TUNING.items():
        cmd += ['--set', f'{key}={value}']
    return cmd


def identity(cases):
    files = [SCRIPT, AUTOTUNE, DTK / 'env.sh', TOOL / 'Auto_tuning.py']
    for sub in ('src', 'source-all', 'source-quick', 'source-test', 'developer_kernels'):
        files += [p for p in (TOOL / sub).rglob('*') if p.is_file() and
                  (p.suffix in ('.py', '.cpp', '.hpp', '.h') or p.name == 'Makefile')]
    for path in files:
        if not path.is_file():
            raise RuntimeError(f'缺少调优工具文件: {path}')
    return {'fingerprint': bench.fingerprint(), 'cases': cases, 'plan': plan(cases),
            'source_sha256': {str(p): sha(p) for p in sorted(set(files))},
            'dtk_root': str(DTK), 'installed_library_sha256': tree_digest(LIBRARY)}


def check_manifest(saved, current):
    if saved != current:
        raise RuntimeError('Workflow configuration fingerprint changed; refusing to mix results')


@contextlib.contextmanager
def environment(**updates):
    missing = object()
    old = {key: os.environ.get(key, missing) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def retry(root, action):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    previous = sorted(root.glob('attempt-*/attempt.json'))
    for path in previous:
        row = read(path)
        if row['status'] == 'success':
            return row['result']
    numbers = [int(p.name.split('-')[-1]) for p in root.glob('attempt-*') if p.is_dir() and p.name.split('-')[-1].isdigit()]
    number = max(numbers, default=0) + 1
    # A previous interruption does not consume a failure; an explicitly resumed exhausted
    # stage receives a fresh pair while preserving all previous attempt directories.
    failures = sum(read(p)['status'] == 'failed' for p in previous)
    budget = 2 if failures >= 2 else 2 - failures
    last = None
    for index in range(number, number + budget):
        directory = root / f'attempt-{index:02d}'
        directory.mkdir()
        try:
            result = action(index, directory)
            save(directory / 'attempt.json', {'attempt': index, 'status': 'success', 'result': result, 'finished_utc': now()})
            return result
        except KeyboardInterrupt:
            save(directory / 'attempt.json', {'attempt': index, 'status': 'interrupted', 'finished_utc': now()})
            raise
        except Exception as exc:
            last = exc
            save(directory / 'attempt.json', {'attempt': index, 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}', 'finished_utc': now()})
            log(f'{directory}: {type(exc).__name__}: {exc}')
            if 'fingerprint changed' in str(exc):
                raise
    raise RuntimeError(f'两次尝试失败: {last}')


def terminate_group(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_process(command, env, log_path):
    global CHILD
    with Path(log_path).open('ab') as stream:
        CHILD = subprocess.Popen(command, cwd=bench.MODEL, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        proc = CHILD
        try:
            code = proc.wait()
            if code:
                raise RuntimeError(f'命令退出码 {code}，查看 {log_path}')
        finally:
            terminate_group(proc)
            CHILD = None


def stop_children():
    terminate_group(CHILD)
    proc = bench.CHILD
    if proc is not None:
        terminate_group(proc)
        bench.CHILD = None


def copy_tree(source, dest):
    if source.exists():
        shutil.copytree(source, dest, dirs_exist_ok=True, symlinks=True)


def snapshot_shared(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in SHARED:
        copy_tree(TOOL / name, destination / name)


def shared_digest():
    files = []
    for name in SHARED:
        folder = TOOL / name
        if folder.is_dir():
            files.extend((p.relative_to(TOOL).as_posix(), sha(p)) for p in sorted(folder.rglob('*')) if p.is_file())
    return hashlib.sha256(json.dumps(files, ensure_ascii=False).encode()).hexdigest()


def installed_unchanged(manifest):
    if tree_digest(LIBRARY) != manifest['installed_library_sha256']:
        raise RuntimeError('Installed library fingerprint changed; refusing to continue')


def capture_attempt(number, directory):
    work = directory / 'work'
    work.mkdir()
    (work / 'runtime.yaml').write_bytes((bench.ROOT / 'configs/runtime.yaml').read_bytes())
    save(directory / 'capture-environment.json', {'ROCBLAS_LAYER': '3', 'ROCBLAS_TENSILE_LIBPATH': None, 'groups': sorted(bench.case_groups(CAPTURE))})
    with environment(ROCBLAS_LAYER='3', ROCBLAS_TENSILE_LIBPATH=None, TENSILE_DB=None, KERNEL_TEST=None):
        config = bench.prepare_case(work, CAPTURE)
        row = bench.run_attempt(work, CAPTURE, 'speed-1', 1, config, bench.fingerprint())
    if row['status'] != 'success':
        raise RuntimeError(row.get('reason', '采集失败'))
    raw = Path(row['directory']) / 'train.log'
    count = clean_layer3(raw, directory / 'layer3.clean.log')
    return {'raw_log': str(raw), 'input': str(directory / 'layer3.clean.log'),
            'input_sha256': sha(directory / 'layer3.clean.log'), 'gemm_commands': count,
            'training': row}


def tune_attempt(number, directory, capture, manifest, root):
    input_path = Path(capture['input'])
    if sha(input_path) != capture['input_sha256']:
        raise RuntimeError('Capture input fingerprint changed')
    installed_unchanged(manifest)
    if number > 1:
        previous = [p for p in sorted(directory.parent.glob('attempt-*')) if p != directory and (p / 'launch.json').exists()]
        if previous:
            recorded = previous[-1] / 'shared-state.sha256'
            if not recorded.is_file() or recorded.read_text().strip() != shared_digest():
                raise RuntimeError('DTK shared work fingerprint changed or previous run was interrupted before recording it')
        snapshot_shared(directory / 'before_retry')
    bench.resource_wait.wait_for_idle_devices(directory / 'resource-wait.jsonl', device_count=8)
    installed_unchanged(manifest)
    check_manifest(manifest, identity(manifest['cases']))
    cmd = tuning_command(input_path)
    save(directory / 'launch.json', {'command': cmd, 'DTKROOT': str(DTK), 'input_sha256': capture['input_sha256'], 'tuning': TUNING})
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('ROCBLAS_LOG_') or key in ('ROCBLAS_LAYER', 'ROCBLAS_TENSILE_LIBPATH', 'TENSILE_DB', 'KERNEL_TEST', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
            env.pop(key, None)
    env['DTKROOT'] = str(DTK)
    shell = ['bash', '-c', 'set -e; source "$1"; shift; exec "$@"', 'gemm', str(DTK / 'env.sh'), *cmd]
    try:
        run_process(shell, env, directory / 'tuning.log')
    finally:
        (directory / 'shared-state.sha256').write_text(shared_digest() + '\n')
        snapshot_shared(directory / 'after_tuning')
        installed_unchanged(manifest)
    if '所有流程执行完毕' not in (directory / 'tuning.log').read_text(errors='replace'):
        raise RuntimeError('选优日志未确认完成')
    generated = TOOL / 'optimization_configs/new/config/library_gpu5'
    if not generated.is_dir() or not list(generated.glob('G*.dat')):
        raise RuntimeError('选优没有生成 G*.dat 配置')
    target = directory / 'selected_config/library_gpu5'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(generated, target)
    installed_unchanged(manifest)
    return {'config': str(target), 'config_sha256': tree_digest(target), 'input_sha256': capture['input_sha256']}


def benchmark_report(root, cases, rows):
    root.mkdir(parents=True, exist_ok=True)
    bench.write_report(root, cases, rows)
    report_path = root / 'report.md'
    text = report_path.read_text()
    text = text.replace('严格兼容基线不启用 pad_156。', '本次所有方案均启用 pad_156；baseline 为兼容项＋pad_156。')
    text = text.replace('Trace 独立采集，不参与测速统计。', '本次不采集 Trace。')
    start = text.find('## Trace\n')
    end = text.find('## 方案 Group\n', start)
    if start != -1 and end != -1:
        text = text[:start] + '## Trace\n\n本次不采集 Trace。\n\n' + text[end:]
    text = text.replace('# PhysAI GR00T batch 960 单项与组合测速', '# PhysAI GR00T GEMM 调优后测速')
    text += '\n所有方案使用同一归档 GEMM 配置；没有未调优对照，不能单独计算 GEMM 收益。\n'
    tmp = report_path.with_name('report.md.tmp')
    tmp.write_text(text)
    tmp.replace(report_path)


def speed_attempt(number, directory, root, case, round_name, config, selected, rows, cases, manifest):
    check_manifest(manifest, identity(cases))
    verify_config(Path(selected['config']), selected['config_sha256'])
    with environment(ROCBLAS_LAYER=None, ROCBLAS_TENSILE_LIBPATH=selected['config'], TENSILE_DB=None, KERNEL_TEST=None):
        current = bench.fingerprint()
        row = bench.run_attempt(root / 'benchmark', case, round_name, number, config, current)
    launch = Path(row['directory']) / 'launch.json'
    if launch.exists():
        launch_data = read(launch)
        launch_data['gemm'] = {'ROCBLAS_TENSILE_LIBPATH': selected['config'], 'config_sha256': selected['config_sha256']}
        save(launch, launch_data)
    if row['status'] == 'success':
        verify_config(Path(selected['config']), selected['config_sha256'])
        check_manifest(manifest, identity(cases))
    save(directory / 'training-result.json', row)
    rows.append(row)
    benchmark_report(root / 'benchmark', cases, rows)
    if row['status'] != 'success':
        raise RuntimeError(row.get('reason', '测速失败'))
    return {'training': row, 'config_sha256': selected['config_sha256']}


def load_speed_rows(root):
    rows = []
    for folder in sorted((root / 'attempts').glob('*/*/attempt-*')):
        if not folder.is_dir():
            continue
        record = folder / 'attempt.json'
        training = folder / 'training-result.json'
        if not record.is_file():
            # Orphan training output was not committed by the orchestrator.
            continue
        status = read(record)
        if status['status'] == 'success':
            row = status.get('result', {}).get('training')
            if not row or row.get('status') != 'success':
                raise RuntimeError(f'Incomplete successful attempt record: {record}')
        elif training.is_file():
            row = read(training)
            row['status'] = status['status']
            row['reason'] = status.get('reason', status['status'])
        else:
            row = {'case': folder.parent.parent.name, 'round': folder.parent.name,
                   'attempt': status['attempt'], 'status': status['status'],
                   'directory': str(folder), 'reason': status.get('reason', status['status'])}
        rows.append(row)
    return rows


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def alive(record):
    return record.get('start') is not None and process_start(record['pid']) == record['start']


@contextlib.contextmanager
def workflow_locks():
    paths = [bench.DATA_ROOT / 'outputs/.gemm-workflow.lock', bench.DATA_ROOT / 'outputs/.physai-groot-benchmark.lock']
    with contextlib.ExitStack() as stack:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open('a+'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'已有测速或 GEMM 编排持有锁: {path}') from exc
        yield


def workflow(root, cases, resume):
    with workflow_locks():
        if resume:
            if not (root / 'manifest.json').is_file():
                raise RuntimeError('续跑目录缺少 manifest.json')
        else:
            root.mkdir(parents=True, exist_ok=False)
        save(root / 'process.json', {'pid': os.getpid(), 'start': process_start(os.getpid())})
        state = read(root / 'state.json') if (root / 'state.json').exists() else {'completed': []}
        state.update(status='checking')
        state.pop('error', None)
        (root / 'exit.code').unlink(missing_ok=True)
        save(root / 'state.json', state)
        try:
            current = identity(cases)
            if resume:
                check_manifest(read(root / 'manifest.json'), current)
            else:
                save(root / 'manifest.json', current)
            manifest = current
            state['status'] = 'running'
            save(root / 'state.json', state)
            benchmark_report(root / 'benchmark', cases, load_speed_rows(root))
            with environment(ROCBLAS_LAYER=None, ROCBLAS_TENSILE_LIBPATH=None, TENSILE_DB=None, KERNEL_TEST=None):
                capture = retry(root / 'capture', capture_attempt)
            check_manifest(manifest, identity(cases))
            if sha(capture['input']) != capture['input_sha256']:
                raise RuntimeError('Capture input fingerprint changed')
            state['capture'] = capture
            state['stage'] = 'tune'
            save(root / 'state.json', state)
            backup = root / 'before_tuning'
            if not (root / 'before_tuning.complete.json').exists():
                snapshot_shared(backup)
                save(root / 'before_tuning.complete.json', {'completed_utc': now()})
            selected = retry(root / 'tuning', lambda n, d: tune_attempt(n, d, capture, manifest, root))
            check_manifest(manifest, identity(cases))
            verify_config(Path(selected['config']), selected['config_sha256'])
            if selected['input_sha256'] != capture['input_sha256']:
                raise RuntimeError('Capture input fingerprint changed for tuning')
            installed_unchanged(manifest)
            state['selected'] = selected
            state['stage'] = 'benchmark'
            save(root / 'state.json', state)
            output = root / 'benchmark'
            output.mkdir(exist_ok=True)
            runtime = output / 'runtime.yaml'
            expected_runtime = (bench.ROOT / 'configs/runtime.yaml').read_bytes()
            if runtime.exists() and runtime.read_bytes() != expected_runtime:
                raise RuntimeError('Runtime configuration fingerprint changed')
            if not runtime.exists():
                runtime.write_bytes(expected_runtime)
            rows = load_speed_rows(root)
            benchmark_report(output, cases, rows)
            for case in cases:
                if identity(cases) != manifest:
                    raise RuntimeError('Workflow configuration fingerprint changed during run')
                config = bench.prepare_case(output, case)
                for round_name in ('speed-1', 'speed-2'):
                    state.update(case=case, round=round_name)
                    save(root / 'state.json', state)
                    retry(root / 'attempts' / case / round_name,
                          lambda n, d: speed_attempt(n, d, root, case, round_name, config, selected, rows, cases, manifest))
            state.update(status='complete', stage='complete')
            save(root / 'state.json', state)
            (root / 'exit.code').write_text('0\n')
            log(f'完成，报告：{output / "report.md"}')
        except BaseException as exc:
            stop_children()
            if (root / 'benchmark').is_dir():
                try:
                    benchmark_report(root / 'benchmark', cases, load_speed_rows(root))
                except Exception as report_error:
                    log(f'报告更新失败：{report_error}')
            state.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            save(root / 'state.json', state)
            (root / 'exit.code').write_text('130\n' if isinstance(exc, KeyboardInterrupt) else '1\n')
            raise


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'收到信号 {signum}')


def start_background(root, cases, resume):
    if root.exists() and not resume:
        raise RuntimeError('输出目录已存在；续跑请使用 resume')
    if resume and not (root / 'manifest.json').is_file():
        raise RuntimeError('续跑目录缺少 manifest.json')
    if (root / 'process.json').exists() and alive(read(root / 'process.json')):
        raise RuntimeError('该输出目录已有运行进程')
    root.parent.mkdir(parents=True, exist_ok=True)
    logfile = root.with_name(root.name + '.log')
    cmd = ['bash', str(SCRIPT), '_run', '--output', str(root), '--cases', *cases]
    if resume:
        cmd.append('--resume')
    env = {**os.environ, 'PHYS_GEMM_PYTHON': sys.executable}
    with logfile.open('ab') as stream:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError(f'后台启动失败，查看 {logfile}')
        if (root / 'process.json').exists() and read(root / 'process.json').get('pid') == proc.pid:
            print(f'结果目录：{root}\nPID：{proc.pid}\n调度日志：{logfile}\n退出码：{root / "exit.code"}', flush=True)
            return
        time.sleep(.1)
    raise RuntimeError(f'尚未确认启动，查看 {logfile}，请勿重复启动')


def main(argv=None):
    parser = argparse.ArgumentParser(description='PhysAI GR00T GEMM 选优与两轮顺序测速')
    parser.add_argument('action', choices=('plan', 'start', 'status', 'stop', 'resume', '_run'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--resume', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(plan(args.cases), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--output 必填')
    root = args.output.expanduser().resolve()
    if args.action in ('status', 'stop'):
        record = read(root / 'process.json') if (root / 'process.json').exists() else {}
        if args.action == 'status':
            print(json.dumps({'running': alive(record), 'state': read(root / 'state.json') if (root / 'state.json').exists() else None,
                              'exit_code': (root / 'exit.code').read_text().strip() if (root / 'exit.code').exists() else None}, ensure_ascii=False, indent=2))
        elif alive(record):
            os.kill(record['pid'], signal.SIGTERM)
            log('已请求停止，请用 status 确认退出')
        else:
            log('该运行没有存活的编排进程')
        return
    if args.action == 'resume':
        manifest = read(root / 'manifest.json')
        if args.cases and selected_cases(args.cases) != manifest['cases']:
            raise RuntimeError('续跑方案列表与原实验不同')
        start_background(root, manifest['cases'], True)
        return
    cases = selected_cases(args.cases)
    if args.action == 'start':
        start_background(root, cases, False)
        return
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with environment(ROCBLAS_LAYER=None, ROCBLAS_TENSILE_LIBPATH=None, TENSILE_DB=None, KERNEL_TEST=None,
                     GR00T_DIT_SDPA_MODE=None, TURBO_PHYSAI_DISABLE_TORCH_COMPILE=None):
        workflow(root, cases, args.resume)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('已中断')
        sys.exit(130)
    except Exception as exc:
        log(f'失败：{exc}')
        sys.exit(1)
GEMM_WORKFLOW_PY
