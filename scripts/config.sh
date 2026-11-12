#!/usr/bin/env bash
# Shared paths; environment variables override these defaults.
ORIENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ORIENT_ROOT"

PYTHON="${PYTHON:-python}"
if [[ -n "${LLAVA_SOURCE:-}" ]]; then
  export PYTHONPATH="$LLAVA_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
fi

DATA_PATH="${DATA_PATH:-data/train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-data/images}"
METADATA="${METADATA:-data/metadata.json}"
REFERENCE_CACHE="${REFERENCE_CACHE:-data/reference.json}"
MODEL_PATH="${MODEL_PATH:-checkpoints/llava-initial}"
REFERENCE_MODEL="${REFERENCE_MODEL:-checkpoints/xi}"
VISION_TOWER="${VISION_TOWER:-checkpoints/clip}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/orient-run}"

SEED="${SEED:-42}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
