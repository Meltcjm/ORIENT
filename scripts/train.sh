#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

# Keep ORIENT's --n out of torchrun's abbreviated option parser.
exec "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
  -m -- orient.train \
  --data-path "$DATA_PATH" --metadata "$METADATA" \
  --reference-cache "$REFERENCE_CACHE" \
  --model-name-or-path "$MODEL_PATH" \
  --reference-model "$REFERENCE_MODEL" --vision-tower "$VISION_TOWER" \
  --image-folder "$IMAGE_FOLDER" --out "$OUTPUT_DIR" \
  --rho "${RHO:-0.2}" --B "${B:-32}" --n "${N:-2}" --seed "$SEED" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
  --group_by_modality_length True \
  "$@"
