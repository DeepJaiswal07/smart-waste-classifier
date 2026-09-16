#!/usr/bin/env bash
# Shell script to run inference on single image
set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/predict.sh <image_path> [top_k]"
    exit 1
fi

IMAGE_PATH="$1"
TOP_K="${2:-3}"

echo "Predicting category for $IMAGE_PATH..."
python -m src.main predict --image "$IMAGE_PATH" --top-k "$TOP_K"
