#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=open/training python open/training/train_fast_1s.py \
  --model-path /path/to/qwen-omni \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir /path/to/outputs/fast_axis \
  --labels neutral,anger,anxiety,sadness,joy,surprise,embarrassment \
  --video-key video_path \
  --prefix-step-sec 1.0 \
  --min-prefix-sec 1.5 \
  --include-full-prefix true \
  --prefix-loss-weighting observation_ratio \
  --num-train-epochs 10 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 2e-5
