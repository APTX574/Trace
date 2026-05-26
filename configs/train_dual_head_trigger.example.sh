#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=open/src python open/scripts/train_dual_head_trigger.py \
  --train-trajectory-jsonl /path/to/train_trajectory.jsonl \
  --train-slow-jsonl /path/to/train_slow_outputs.jsonl \
  --eval-trajectory-jsonl /path/to/dev_trajectory.jsonl \
  --eval-slow-jsonl /path/to/dev_slow_outputs.jsonl \
  --output-dir /path/to/outputs/dual_head_trigger \
  --decision-mode threshold \
  --epochs 300 \
  --lr 1e-3
