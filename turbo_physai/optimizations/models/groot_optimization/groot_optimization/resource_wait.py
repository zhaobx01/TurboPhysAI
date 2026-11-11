"""Wait for the benchmark's HCU devices to remain below a VRAM threshold."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time


def _query_devices():
    result = subprocess.run(
        ['rocm-smi', '--showmeminfo', 'vram', '--json'],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return json.loads(result.stdout)


def _memory_percentages(cards, device_count):
    percentages = {}
    for rank in range(device_count):
        name = f'card{rank}'
        card = cards[name]
        unit = 'B' if 'vram Total Used Memory (B)' in card else 'MiB'
        used = float(card[f'vram Total Used Memory ({unit})'])
        total = float(card[f'vram Total Memory ({unit})'])
        if not math.isfinite(used) or not math.isfinite(total) or total <= 0 or not 0 <= used <= total:
            raise ValueError(f'{name} 的显存数据无效')
        percentages[name] = 100 * used / total
    return percentages


def wait_for_idle_devices(log_path, *, device_count=8, query=_query_devices, sleep=time.sleep, clock=time.monotonic):
    """Require three idle samples and report total elapsed waiting time."""
    consecutive = 0
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = clock()
    while True:
        row = {'time_utc': datetime.now(timezone.utc).isoformat()}
        try:
            percentages = _memory_percentages(query(), device_count)
            row['memory_percent'] = percentages
            row['idle'] = all(value < 5 for value in percentages.values())
            consecutive = consecutive + 1 if row['idle'] else 0
            row['max_memory_percent'] = max(percentages.values())
            status = f"最高显存占用 {row['max_memory_percent']:.2f}%"
        except Exception as exc:
            consecutive = 0
            row['idle'] = False
            row['error'] = f'{type(exc).__name__}: {exc}'
            status = f"显存查询失败：{row['error']}"
        row['consecutive_idle'] = consecutive
        row['elapsed_seconds'] = clock() - start
        minutes, seconds = divmod(int(row['elapsed_seconds']), 60)
        with path.open('a', encoding='utf-8') as log:
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'{status}；连续空闲 {consecutive}/3；已等待 {minutes}分{seconds:02d}秒', flush=True)
        if consecutive == 3:
            return
        sleep(10)


def main():
    parser = argparse.ArgumentParser(description='等待 HCU 显存连续空闲后启动测速')
    parser.add_argument('--log', required=True, help='JSONL 资源等待日志')
    parser.add_argument('--devices', type=int, default=8, help='需要检查的 card 数量')
    args = parser.parse_args()
    if args.devices < 1:
        parser.error('--devices 必须大于零')
    wait_for_idle_devices(args.log, device_count=args.devices)


if __name__ == '__main__':
    main()
