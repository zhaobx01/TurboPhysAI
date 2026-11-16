#!/usr/bin/env bash
set -euo pipefail
# 每 35 秒检查全部设备，连续四次显存占用低于 5% 后启动。
export GR00T_BENCH_IDLE_INTERVAL=35 GR00T_BENCH_IDLE_SAMPLES=4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --worker ]]; then
    shift
    output=$1
    shift
    child=''
    terminate() { [[ -z $child ]] || kill -TERM "$child" 2>/dev/null || true; }
    trap terminate TERM INT
    python3 -u "$SCRIPT_DIR/benchmark.py" --output "$output" "$@" &
    child=$!
    set +e
    wait "$child"
    code=$?
    if kill -0 "$child" 2>/dev/null; then wait "$child"; code=$?; fi
    printf '%s\n' "$code" > "$output.exit"
    exit "$code"
fi
output=${GR00T_BENCH_OUTPUT:-/workspace/data/outputs/physai-groot-b960-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$(dirname "$output")"
setsid nohup bash "$SCRIPT_DIR/run_benchmark_b960.sh" --worker "$output" "$@" >"$output.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output.pid"
printf '结果目录：%s\nPID：%s\n调度日志：%s.log\n退出码：%s.exit\n' "$output" "$!" "$output" "$output"
