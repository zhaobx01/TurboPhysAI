#!/usr/bin/env python3
"""Reproducible PhysAI GR00T performance experiments."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
PHYS_ROOT = ROOT.parents[3]
MODEL = Path(os.environ.get('GROOT_MODEL', '/workspace/TurboPhysAI/model/Isaac-GR00T-official')).resolve()
DATA_ROOT = Path('/workspace/data')
SHA = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PHYS_ROOT))
import groot_optimization.catalog  # noqa: E402,F401
import resource_wait  # noqa: E402
from turbo_physai.engine.definitions.registry import default_registry  # noqa: E402

BASELINE = {'groot.compat.training_config', 'groot.dit.hip_math_sdpa'}
SINGLES = {
    'pad_156': 'groot.data.pad_156',
    'qwen_fast_pos': 'groot.qwen.fast_pos_embed_interpolate',
    'qwen_rot_pos': 'groot.qwen.rot_pos_emb',
    'qwen_rope_index': 'groot.qwen.get_rope_index',
    'vision_block': 'groot.compile.qwen_vision_block',
    'dit_model': 'groot.compile.dit',
    'vl': 'groot.compile.vl',
    'pixel_bf16': 'groot.data.pixel_values_bf16',
    'fused_adam': 'groot.optim.apex_fused_adam',
    'gro': 'groot.zero2.grad_reduce_overlap',
    'text_forwards': 'groot.compile.qwen_text_layer',
    'obr': 'groot.zero2.owner_reduce',
    'allgather': 'groot.zero2.allgather_pipeline',
}
CASES = {'baseline': set(), **{name: {group} for name, group in SINGLES.items()}}
CASES['combo_runtime'] = {SINGLES[n] for n in ('pixel_bf16', 'fused_adam', 'gro')}
CASES['combo_dit'] = CASES['combo_runtime'] | {SINGLES['dit_model']}
CASES['combo_text_forwards'] = CASES['combo_runtime'] | {SINGLES['text_forwards']}
CASES['combo_obr'] = CASES['combo_runtime'] | {SINGLES['obr']}
CASES['combo_obr_dit_vl'] = CASES['combo_obr'] | {SINGLES['dit_model'], SINGLES['vl']}
CASES['combo_obr_dit_vl_text'] = CASES['combo_obr_dit_vl'] | {SINGLES['text_forwards']}
CASES['combo_obr_dit_vl_text_allgather'] = CASES['combo_obr_dit_vl_text'] | {SINGLES['allgather']}
CASES['all_on'] = set(SINGLES.values())


def case_groups(case):
    groups = set(BASELINE | CASES[case])
    todo = list(groups)
    while todo:
        for dep in default_registry.get_group(todo.pop()).depends_on:
            if dep not in groups:
                groups.add(dep)
                todo.append(dep)
    return groups


def measure(records):
    if len(records) != 8 or {x.get('rank') for x in records} != set(range(8)):
        raise ValueError('Expected exactly eight distinct rank records')
    if any(x.get('world_size') != 8 or len(x.get('step_seconds', [])) != 40 or
           any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in x['step_seconds']) for x in records):
        raise ValueError('Expected 40 positive finite timings for every rank')
    steps = [max(x['step_seconds'][i] for x in records) for i in range(40)]
    steady = steps[10:40]
    avg = statistics.mean(steps[30:40])
    return {'valid': True, 'step35_ms': steps[34] * 1000,
            'step35_samples_per_second': 960 / steps[34], 'mean31_40_ms': avg * 1000,
            'mean31_40_samples_per_second': 960 / avg,
            'cv_pct': statistics.pstdev(steady) / statistics.mean(steady) * 100,
            'drift_pct': (statistics.mean(steady[15:]) / statistics.mean(steady[:15]) - 1) * 100,
            'max_memory_allocated_bytes': max(x.get('max_memory_allocated_bytes', 0) for x in records),
            'max_memory_reserved_bytes': max(x.get('max_memory_reserved_bytes', 0) for x in records)}


def next_attempt(rows, case, round_name, retry_failed=False):
    selected = [r for r in rows if r['case'] == case and r['round'] == round_name]
    if any(r['status'] == 'success' for r in selected):
        return None
    if sum(r['status'] == 'failed' for r in selected) >= 2 and not retry_failed:
        return None
    return max((r['attempt'] for r in selected), default=0) + 1


def aggregate(rows, case):
    rounds = []
    for name in ('speed-1', 'speed-2'):
        success = [x for x in rows if x['case'] == case and x['round'] == name and x['status'] == 'success']
        if success:
            rounds.append(success[-1])
    out = {'successful_rounds': len(rounds), 'label': ('无成功轮次', '仅一轮', '两轮中位数')[len(rounds)]}
    for k in ('step35_ms', 'step35_samples_per_second', 'mean31_40_ms', 'mean31_40_samples_per_second'):
        out[k] = statistics.median(x[k] for x in rounds) if rounds else None
    return out


def validate_trace(attempt):
    path = Path(attempt) / 'profiling'
    for rank in range(8):
        trace = path / f'trace_rank_{rank}_iter_35.json'
        marker = path / f'profiler-stopped-rank-{rank}.json'
        if not trace.is_file() or trace.stat().st_size == 0 or not marker.is_file():
            raise ValueError(f'Missing trace or profiler stop marker for rank {rank}')
        with trace.open() as trace_file:
            trace_data = json.load(trace_file)
        if not isinstance(trace_data.get('traceEvents'), list):
            raise ValueError(f'Invalid Chrome trace for rank {rank}')
        del trace_data
        if json.loads(marker.read_text()) != {'rank': rank, 'step': 35, 'stopped_after_trace': True}:
            raise ValueError(f'Invalid profiler stop marker for rank {rank}')
    if not (path / 'rank0-key-averages.txt').is_file() or not (path / 'rank0-key-averages.txt').stat().st_size:
        raise ValueError('Missing rank 0 operator summary')
    return {'trace_step': 35}


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def git_env():
    env = os.environ.copy()
    count = int(env.get('GIT_CONFIG_COUNT', '0'))
    env[f'GIT_CONFIG_KEY_{count}'] = 'safe.directory'
    env[f'GIT_CONFIG_VALUE_{count}'] = str(MODEL)
    env['GIT_CONFIG_COUNT'] = str(count + 1)
    return env


def fingerprint():
    env = git_env()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(MODEL), *args], env=env, text=True).strip()
    if git('rev-parse', 'HEAD') != SHA or git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Official model must be clean and at ' + SHA)
    import importlib.metadata
    versions = {}
    for package in ('torch', 'transformers', 'deepspeed', 'accelerate', 'wandb'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = [ROOT / 'configs/recipe.yaml', ROOT / 'configs/runtime.yaml', Path(__file__), ROOT / 'scripts/rank_launcher.py', ROOT / 'scripts/resource_wait.py', ROOT / 'scripts/run_benchmark_b960.sh']
    files += [PHYS_ROOT / 'turbo_physai' / name for name in ('__init__.py', 'cli.py', 'runner.py', 'runtime.py', 'compatibility.py')]
    files += sorted((PHYS_ROOT / 'turbo_physai/bootstrap').rglob('*.py'))
    files += sorted((ROOT / 'groot_optimization').rglob('*.py'))
    files += sorted((PHYS_ROOT / 'turbo_physai/engine').rglob('*.py'))
    files += sorted((PHYS_ROOT / 'turbo_physai/cli').rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(PHYS_ROOT)).encode() + b'\0' + path.read_bytes() + b'\0')
    environment_keys = ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'OMP_NUM_THREADS', 'DISABLE_ADDMM_CUDA_LT', 'VLM_MODEL_PATH', 'VIDEO_DECODER_BACKEND', 'USE_WANDB')
    prefixes = ('GR00T_', 'TURBO_PHYSAI_', 'TORCH', 'PYTORCH_', 'CUDA_', 'HIP_', 'HSA_', 'ROCM_', 'ROCR_', 'NCCL_', 'RCCL_', 'MIOPEN_', 'ROCBLAS_', 'TRANSFORMERS_', 'HF_')
    # Hash potentially sensitive values; retain only names for auditability.
    keys = sorted(key for key in os.environ if key in environment_keys or key.startswith(prefixes))
    environment = {'keys': keys, 'sha256': hashlib.sha256(json.dumps({key: os.environ[key] for key in keys}, sort_keys=True).encode()).hexdigest()}
    return {'model': str(MODEL), 'model_sha': SHA, 'source_sha256': digest.hexdigest(),
            'python': sys.version, 'versions': versions, 'environment': environment,
            'training': {'gpus': 8, 'global_batch': 960, 'gradient_accumulation': 1, 'steps': 40,
                         'learning_rate': 1e-4, 'weight_decay': 1e-5, 'warmup_ratio': .05,
                         'workers': 4, 'dataset': str(DATA_ROOT / 'libero_10_no_noops_1.0.0_lerobot_h264'),
                         'base_model': str(DATA_ROOT / 'checkpoints/GR00T-N1.7-3B-local-cosmos'),
                         'embodiment': 'LIBERO_PANDA', 'shard_size': 1024, 'shards_per_epoch': 100000,
                         'episode_sampling_rate': .1, 'state_dropout': .2, 'jitter': [.3, .4, .5, .08]}}


def prepare_case(output, case):
    folder = output / case / 'configs'
    folder.mkdir(parents=True, exist_ok=True)
    recipe = yaml.safe_load((ROOT / 'configs/recipe.yaml').read_text())
    enabled = case_groups(case)
    for group in recipe['optimization_groups']:
        group['enabled'] = group['id'] in enabled
    recipe['metadata']['id'] = 'model.groot.benchmark.' + case.replace('_', '-')
    recipe_path = folder / 'recipe.yaml'
    config = folder / 'optimization.yaml'
    env = git_env()
    env['PYTHONPATH'] = os.pathsep.join([str(PHYS_ROOT), str(ROOT), str(MODEL), env.get('PYTHONPATH', '')])
    if config.exists():
        if not recipe_path.is_file() or yaml.safe_load(recipe_path.read_text()) != recipe:
            raise RuntimeError(f'Saved case recipe changed for {case}; refusing to mix results')
        actual = {g['id'] for g in yaml.safe_load(config.read_text())['optimization_groups'] if g.get('enabled')}
        if actual != enabled:
            raise RuntimeError(f'Saved case configuration changed for {case}; refusing to mix results')
    else:
        recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
        cmd = [sys.executable, '-m', 'turbo_physai.cli', 'optimization', 'generate', '--recipe', str(recipe_path),
               '--repo', str(MODEL), '--commit', SHA, '--output', str(config), '--force']
        result = subprocess.run(cmd, cwd=MODEL, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (folder / 'generate.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'Config generation failed for {case}: {result.stdout}')
    check = subprocess.run([sys.executable, '-m', 'turbo_physai.cli', 'optimization', 'check', str(config), '--repo', str(MODEL)],
                           cwd=MODEL, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (folder / 'check.log').write_text(check.stdout)
    if check.returncode:
        raise RuntimeError(f'Config check failed for {case}: {check.stdout}')
    snapshot_path = folder / 'snapshot.sha256'
    digest = case_snapshot_digest(config)
    if snapshot_path.exists() and snapshot_path.read_text().strip() != digest:
        raise RuntimeError(f'Case configuration fingerprint changed for {case}')
    if not snapshot_path.exists():
        snapshot_path.write_text(digest + '\n')
    return config


def case_snapshot_digest(config):
    folder = config.parent
    digest = hashlib.sha256()
    for name in ('recipe.yaml', 'optimization.yaml', '.optimization.yaml.generation.json'):
        path = folder / name
        digest.update(name.encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def verify_snapshots(output, config):
    if (output / 'runtime.yaml').read_bytes() != (ROOT / 'configs/runtime.yaml').read_bytes():
        raise RuntimeError('Runtime configuration fingerprint changed; refusing to mix results')
    if (config.parent / 'snapshot.sha256').read_text().strip() != case_snapshot_digest(config):
        raise RuntimeError('Case configuration fingerprint changed; refusing to mix results')


def available_port():
    with socket.socket() as sock:
        sock.bind(('', 0))
        return str(sock.getsockname()[1])


def train_command(config, attempt, runtime):
    data = DATA_ROOT
    return [sys.executable, '-m', 'turbo_physai.cli', 'run', '--disable-numa', '--log-report',
            '--runtime-config', str(runtime), '--optimization-config', str(config), '--',
            'torchrun', '--nproc_per_node=8', '--master_port=' + available_port(), str(ROOT / 'scripts/rank_launcher.py'),
            '--base_model_path', str(data / 'checkpoints/GR00T-N1.7-3B-local-cosmos'),
            '--dataset_path', str(data / 'libero_10_no_noops_1.0.0_lerobot_h264'), '--embodiment_tag', 'LIBERO_PANDA',
            '--num_gpus', '8', '--output_dir', str(attempt / 'out'), '--global_batch_size', '960',
            '--gradient_accumulation_steps', '1', '--dataloader_num_workers', '4', '--max_steps', '40',
            '--save_steps', '1000', '--save_total_limit', '5', '--learning_rate', '1e-4',
            '--weight_decay', '1e-5', '--warmup_ratio', '0.05', '--shard_size', '1024',
            '--num_shards_per_epoch', '100000', '--episode_sampling_rate', '0.1', '--state_dropout_prob', '0.2',
            '--color_jitter_params', 'brightness', '0.3', 'contrast', '0.4', 'saturation', '0.5', 'hue', '0.08']


def monitor_memory(path, stop):
    while not stop.is_set():
        row = {'time_utc': datetime.now(timezone.utc).isoformat()}
        try:
            cards = resource_wait._query_devices()
            row['cards'] = cards
            row['memory_percent'] = resource_wait._memory_percentages(cards, 8)
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        with path.open('a') as out:
            out.write(json.dumps(row) + '\n')
        stop.wait(10)


CHILD = None

def stop_child():
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
            CHILD.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait()
        except ProcessLookupError:
            pass


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def run_attempt(output, case, round_name, number, config, expected_fingerprint):
    global CHILD
    attempt = output / case / round_name / f'attempt-{number:02d}'
    attempt.mkdir(parents=True, exist_ok=False)
    row = {'case': case, 'round': round_name, 'attempt': number, 'directory': str(attempt),
           'started_utc': datetime.now(timezone.utc).isoformat()}
    stop = threading.Event()
    sampler = None
    try:
        if fingerprint() != expected_fingerprint:
            raise RuntimeError('Configuration fingerprint changed; refusing to mix results')
        verify_snapshots(output, config)
        resource_wait.wait_for_idle_devices(
            attempt / 'resource-wait.jsonl', device_count=8,
            check_interval=float(os.environ.get('GR00T_BENCH_IDLE_INTERVAL', '30')),
            required_consecutive=int(os.environ.get('GR00T_BENCH_IDLE_SAMPLES', '3')))
        if fingerprint() != expected_fingerprint:
            raise RuntimeError('Configuration fingerprint changed; refusing to mix results')
        verify_snapshots(output, config)
        cmd = train_command(config, attempt, output / 'runtime.yaml')
        env = git_env()
        env.update(GROOT_MODEL=str(MODEL), PHYS_BENCH_ATTEMPT_DIR=str(attempt),
                   PHYS_BENCH_TRACE='1' if round_name == 'trace' else '0',
                   PYTHONPATH=os.pathsep.join([str(PHYS_ROOT), str(ROOT), str(MODEL), os.environ.get('PYTHONPATH', '')]))
        env.pop('GR00T_DIT_SDPA_MODE', None)
        env.pop('TURBO_PHYSAI_DISABLE_TORCH_COMPILE', None)
        atomic_json(attempt / 'launch.json', {'command': cmd, 'case': case, 'round': round_name,
                    'groups': sorted(case_groups(case)), 'model_sha': SHA,
                    'environment_overrides': {'GROOT_MODEL': str(MODEL), 'PHYS_BENCH_TRACE': env['PHYS_BENCH_TRACE']}})
        sampler = threading.Thread(target=monitor_memory, args=(attempt / 'device-memory.jsonl', stop), daemon=True)
        sampler.start()
        print(f'开始 {case} {round_name} 第 {number} 次尝试：{attempt}', flush=True)
        with (attempt / 'train.log').open('w') as log:
            CHILD = subprocess.Popen(cmd, cwd=MODEL, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            rc = CHILD.wait()
            CHILD = None
        if rc:
            raise RuntimeError(f'Training exited with code {rc}; see train.log')
        records = [json.loads((attempt / 'step_metrics' / f'rank-{rank:02d}.json').read_text()) for rank in range(8)]
        metrics = measure(records)
        files = list((attempt / 'out').rglob('checkpoint-*')) + list((attempt / 'out').rglob('*.safetensors')) + list((attempt / 'out').rglob('*.bin'))
        if files:
            raise RuntimeError(f'Unexpected saved checkpoint or weights: {files[0]}')
        logtext = (attempt / 'train.log').read_text(errors='replace')
        applied = set(re.findall(r'^\s*- (groot\.[\w.]+): status=applied\b', logtext, re.MULTILINE))
        if applied != case_groups(case):
            raise RuntimeError(f'Applied Group mismatch: expected {sorted(case_groups(case))}; got {sorted(applied)}')
        samples = [json.loads(line) for line in (attempt / 'device-memory.jsonl').read_text().splitlines()]
        row['device_peak_percent'] = {f'card{i}': max((sample['memory_percent'][f'card{i}'] for sample in samples if 'memory_percent' in sample), default=None) for i in range(8)}
        if round_name == 'trace':
            row.update({key: metrics[key] for key in ('max_memory_allocated_bytes', 'max_memory_reserved_bytes')})
            row.update(validate_trace(attempt))
        else:
            row.update(metrics)
        row['status'] = 'success'
    except KeyboardInterrupt:
        stop_child()
        row.update(status='interrupted', reason='user interrupted')
        raise
    except Exception as exc:
        row.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        if 'fingerprint changed' in str(exc):
            row['fatal'] = True
        print(f'{case} {round_name} attempt {number}: {row["reason"]}', flush=True)
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=25)
        row['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(attempt / 'attempt.json', row)
    return row


def memory_summary(row):
    """Summarize raw device samples without conflating missing data with zero."""
    path = Path(row['directory']) / 'device-memory.jsonl'
    values = []
    success = failures = 0
    if not path.is_file():
        return None, None
    for line in path.read_text().splitlines():
        try:
            sample = json.loads(line)
            if 'error' in sample:
                raise ValueError(sample['error'])
            cards = sample['cards']
            current = []
            for card in cards.values():
                if 'vram Total Used Memory (MiB)' in card:
                    current.append(float(card['vram Total Used Memory (MiB)']))
                elif 'vram Total Used Memory (B)' in card:
                    current.append(float(card['vram Total Used Memory (B)']) / 1048576)
            if not current or any(not math.isfinite(x) or x < 0 for x in current):
                raise ValueError('invalid VRAM sample')
            values.extend(current)
            success += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            failures += 1
    return (max(values) if values else None), f'{success}/{failures}'


def render_report(cases, rows, aggregates):
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'

    def percent(value):
        return '—' if value is None else f'{value:+.2f}%'

    def safe(value):
        return str(value).replace('|', r'\|').replace('\n', ' ')

    lines = [
        '# PhysAI GR00T batch 960 单项与组合测速', '',
        '官方 51d4c89；8 卡、全局 batch 960，每轮 40 步。每步耗时取 8 个 rank 的最大墙钟耗时，不额外同步设备；吞吐量为 960 ÷ 对应耗时。', '',
        '两轮成功时显示中位数；只有一轮成功时显示该轮并标注“仅一轮”。CV 和漂移供参考，不按 CV 或漂移剔除成功轮次。Trace 独立采集，不参与测速统计。', '',
        '严格兼容基线不启用 pad_156。各方案自动补全依赖，完整 Group 见文末；编译缓存使用框架默认共享策略。', '',
        'PhysAI 的 dit 编译 DiT 和 AlternateVLDiT.forward，vl 编译 SelfAttentionTransformer.forward；历史仓库同名方案不自动视为等价。测速不证明数值等价。', '',
    ]
    for title, duration, throughput in (
        ('第 35 步', 'step35_ms', 'step35_samples_per_second'),
        ('第 31–40 步均值', 'mean31_40_ms', 'mean31_40_samples_per_second'),
    ):
        lines.extend([f'## {title}', '', '| 方案 | 成功/计划轮数 | 耗时 ms | 吞吐 samples/s | 相对基线 | 说明 |',
                      '|---|---:|---:|---:|---:|---|'])
        for case in cases:
            a = aggregates[case]
            lines.append(f"| {case} | {a['successful_rounds']}/2 | {fmt(a.get(duration))} | {fmt(a.get(throughput))} | {percent(a.get(throughput + '_improvement_pct'))} | {a['label']} |")
        lines.append('')
    lines.extend(['## 每轮明细', '',
                  '| 方案 / 轮次 | 尝试 | 状态 | 第 35 步 ms | 第 31–40 步均值 ms | 第 35 步吞吐 samples/s | CV % | 漂移 % | 错误 |',
                  '|---|---:|---|---:|---:|---:|---:|---:|---|'])
    for case in cases:
        for name in ('speed-1', 'speed-2'):
            attempts = [r for r in rows if r['case'] == case and r['round'] == name]
            if not attempts:
                lines.append(f'| {case} / {name} | — | 待运行 | — | — | — | — | — | — |')
            for row in attempts:
                lines.append(f"| {case} / {name} | {row['attempt']} | {safe(row['status'])} | {fmt(row.get('step35_ms'))} | {fmt(row.get('mean31_40_ms'))} | {fmt(row.get('step35_samples_per_second'))} | {fmt(row.get('cv_pct'))} | {fmt(row.get('drift_pct'))} | {safe(row.get('reason', ''))} |")
    lines.extend(['', '## 测速显存', '',
                  '设备值为每 10 秒采样的单卡最大占用，包含 PyTorch 外占用，可能漏过瞬时峰值；PyTorch 值为各 rank 分配器峰值的最大值。采样失败不改变测速结果，缺失显示为 —。', '',
                  '| 方案 / 轮次 | 尝试 | 设备采样最大 MiB | PyTorch allocated 峰值 MiB | PyTorch reserved 峰值 MiB | 成功/失败采样数 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for row in rows:
        if row['round'] not in ('speed-1', 'speed-2'):
            continue
        device, counts = memory_summary(row)
        allocated = row.get('max_memory_allocated_bytes')
        reserved = row.get('max_memory_reserved_bytes')
        lines.append(f"| {row['case']} / {row['round']} | {row['attempt']} | {fmt(device)} | {fmt(allocated / 1048576 if allocated is not None else None)} | {fmt(reserved / 1048576 if reserved is not None else None)} | {counts or '—'} |")
    lines.extend(['', '## Trace', '', '| 方案 / 轮次 | 尝试 | 状态 | 数据目录 | 错误 |', '|---|---:|---|---|---|'])
    for case in cases:
        traces = [r for r in rows if r['case'] == case and r['round'] == 'trace']
        if not traces:
            lines.append(f'| {case} / trace | — | 待运行 | — | — |')
        for row in traces:
            lines.append(f"| {case} / trace | {row['attempt']} | {safe(row['status'])} | `{safe(row['directory'])}/profiling` | {safe(row.get('reason', ''))} |")
    lines.extend(['', '## 方案 Group', ''])
    for case in cases:
        lines.append(f"- **{case}**：" + ', '.join(f'`{group}`' for group in sorted(case_groups(case))))
    lines.extend(['', '逐 rank 耗时、显存原始采样、资源等待记录和训练日志保存在各尝试目录；失败尝试均保留。', ''])
    return '\n'.join(lines)


def write_report(output, cases, rows):
    aggregates = {case: aggregate(rows, case) for case in cases}
    baseline = aggregates.get('baseline', {})
    for values in aggregates.values():
        for metric in ('step35_samples_per_second', 'mean31_40_samples_per_second'):
            ref = baseline.get(metric)
            values[metric + '_improvement_pct'] = (values[metric] / ref - 1) * 100 if ref and values[metric] else None
    atomic_json(output / 'results.json', {'cases': cases, 'attempts': rows, 'aggregates': aggregates})
    tmp = output / 'report.md.tmp'
    tmp.write_text(render_report(cases, rows, aggregates))
    os.replace(tmp, output / 'report.md')


def load_attempts(output):
    rows = []
    for path in sorted(output.glob('*/[st]*/attempt-*/attempt.json')):
        rows.append(json.loads(path.read_text()))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='PhysAI GR00T 22-case benchmark')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    cases = list(dict.fromkeys(args.cases))
    if 'baseline' not in cases:
        parser.error('--cases must include baseline for same-batch comparison')
    cases.remove('baseline')
    cases.insert(0, 'baseline')
    if args.dry_run:
        print(json.dumps({'cases': {case: sorted(case_groups(case)) for case in cases},
                          'runs': len(cases) * 3, 'rounds': ['speed-1', 'speed-2', 'trace']}, indent=2))
        return 0
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_ROOT / 'outputs/.physai-groot-benchmark.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another PhysAI GR00T benchmark is already running')
        current = fingerprint()
        manifest_path = args.output / 'manifest.json'
        if args.resume:
            if not manifest_path.is_file():
                raise RuntimeError('Cannot resume without manifest.json')
            manifest = json.loads(manifest_path.read_text())
            if manifest['fingerprint'] != current or manifest['cases'] != cases:
                raise RuntimeError('Configuration fingerprint or selected cases changed; cannot resume')
            if (args.output / 'runtime.yaml').read_bytes() != (ROOT / 'configs/runtime.yaml').read_bytes():
                raise RuntimeError('Saved runtime configuration differs; cannot resume')
        else:
            if manifest_path.exists():
                raise RuntimeError('Output already initialized; use --resume')
            manifest = {'fingerprint': current, 'cases': cases, 'created_utc': datetime.now(timezone.utc).isoformat()}
            atomic_json(manifest_path, manifest)
            (args.output / 'runtime.yaml').write_bytes((ROOT / 'configs/runtime.yaml').read_bytes())
        rows = load_attempts(args.output)
        write_report(args.output, cases, rows)
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            for case in cases:
                if fingerprint() != current:
                    raise RuntimeError('Configuration fingerprint changed; refusing to mix results')
                config = prepare_case(args.output, case)
                for round_name in ('speed-1', 'speed-2', 'trace'):
                    history = [r for r in rows if r['case'] == case and r['round'] == round_name]
                    failures = sum(r['status'] == 'failed' for r in history)
                    number = next_attempt(rows, case, round_name, args.retry_failed)
                    if number is None:
                        continue
                    budget = 2 if failures >= 2 and args.retry_failed else 2 - failures
                    while budget > 0:
                        while (args.output / case / round_name / f'attempt-{number:02d}').exists():
                            number += 1
                        try:
                            row = run_attempt(args.output, case, round_name, number, config, current)
                        except KeyboardInterrupt:
                            rows = load_attempts(args.output)
                            write_report(args.output, cases, rows)
                            raise
                        rows.append(row)
                        write_report(args.output, cases, rows)
                        if row.get('fatal'):
                            raise RuntimeError(row['reason'])
                        if row['status'] == 'success':
                            break
                        number += 1
                        budget -= 1
            failures = [r for r in rows if r['status'] == 'failed']
            print(f'完成，报告：{args.output / "report.md"}；失败尝试：{len(failures)}', flush=True)
            return 0 if all(next_attempt(rows, case, r) is None and any(
                x['case'] == case and x['round'] == r and x['status'] == 'success' for x in rows)
                for case in cases for r in ('speed-1', 'speed-2', 'trace')) else 1
        finally:
            stop_child()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('已中断，子训练进程组已停止。', file=sys.stderr, flush=True)
        sys.exit(130)
