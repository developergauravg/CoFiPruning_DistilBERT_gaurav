#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK="${1:-sst2}"
shift || true

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src" python "$PROJECT_DIR/run_sweep.py" \
  --task "$TASK" \
  --output-dir outputs \
  "$@"
