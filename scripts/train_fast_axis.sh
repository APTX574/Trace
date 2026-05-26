#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/training:${PYTHONPATH:-}"

python "${ROOT_DIR}/training/train_fast_1s.py" "$@"
