#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

for task in sst2 qnli ag_news; do
  PYTHONPATH="$PROJECT_DIR/src" python "$PROJECT_DIR/run_sweep.py" \
    --task "$task" \
    --output-dir outputs \
    "$@"
done
