#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

exec "$PYTHON" -m orient.reference \
  --data-path "$DATA_PATH" \
  --model-name-or-path "$MODEL_PATH" \
  --reference-model "$REFERENCE_MODEL" \
  --vision-tower "$VISION_TOWER" \
  --image-folder "$IMAGE_FOLDER" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  --out "$REFERENCE_CACHE" \
  "$@"
