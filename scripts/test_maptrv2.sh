#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
#
# One-shot test for every MapTRv2 optimization Group.
#
# Runs the whole MapTRv2 optimization suite (catalog structure, per-Group
# implementations, data pipeline, DDP flags) plus the engine test for the
# inherited ``mmcv.msda`` Group, and first audits that no Group listed in the
# OptimizationConfig is left without a test that mentions it.  The audit is
# what makes "all Groups" checkable rather than assumed: a Group added to the
# config but not to the suite fails here instead of passing silently.
#
#   scripts/test_maptrv2.sh                  # everything, pytest defaults
#   scripts/test_maptrv2.sh -x -vv           # extra arguments go to pytest
#   scripts/test_maptrv2.sh -m "not model_deps"
#
# Exits non-zero if the audit or any test fails.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG="turbo_physai/optimizations/models/maptrv2_optimization/configs/optimization.yaml"

echo "== MapTRv2 optimization Groups in $(basename "${CONFIG}") =="

python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
modules = {
    path: path.read_text(encoding="utf-8")
    for path in sorted(Path("test").rglob("*.py"))
}

missing = []
for entry in yaml.safe_load(config_path.read_text(encoding="utf-8"))["optimization_groups"]:
    group_id = entry["id"]
    owners = [path for path, text in modules.items() if group_id in text]
    if owners:
        print(f"  {group_id:<32} {', '.join(str(path) for path in owners)}")
    else:
        missing.append(group_id)

if missing:
    print(
        f"\nuntested optimization Groups: {', '.join(missing)}\n"
        "add them to test/optimizations/ or test/engine/ before trusting this run",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo
echo "== pytest =="

mapfile -t FILES < <(
    printf '%s\n' test/optimizations/test_maptrv2_*.py test/engine/test_builtin_msda.py
)

printf '  %s\n' "${FILES[@]}"
echo

python -m pytest "${FILES[@]}" "$@"
