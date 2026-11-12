#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

exec "$PYTHON" -m orient.prepare \
  --data-path "$DATA_PATH" \
  --feedback-per-task "${FEEDBACK_PER_TASK:-20}" --seed "$SEED" \
  --out "$METADATA" \
  "$@"
