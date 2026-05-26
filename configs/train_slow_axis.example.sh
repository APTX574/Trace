#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=open/training python open/training/train_slow_meld.py \
  --model-path /path/to/qwen-omni \
  --train-states-jsonl /path/to/train_states.jsonl \
  --val-states-jsonl /path/to/val_states.jsonl \
  --test-states-jsonl /path/to/test_states.jsonl \
  --output-dir /path/to/outputs/slow_axis \
  --labels neutral,anger,anxiety,sadness,joy,surprise,embarrassment \
  --video-key video_path \
  --context-mode prev_current_next \
  --require-previous-video true \
  --require-next-video true \
  --slow-output-fields reason,new_summary \
  --num-train-epochs 5 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5
