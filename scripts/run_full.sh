#!/usr/bin/env bash
# Full-scale TSGM C-MAPSS run: data prep -> train (iter_pre=5000/iter_main=10000) ->
# recursive PC sampling (n_steps=1000) -> 10-seed discriminative/predictive/t-SNE evaluation.
# See docs/reproduction_plan.md for the reproduction plan this follows.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=".venv/bin/python"
CONFIG="configs/cmapss.yaml"
CKPT_DIR="outputs/checkpoints/cmapss_full"
DEVICE="${DEVICE:-cpu}"
N_SAMPLES="${N_SAMPLES:-100}"
N_STEPS="${N_STEPS:-1000}"
N_SEEDS="${N_SEEDS:-10}"

if [ ! -x "$PYTHON" ]; then
  echo "== creating venv and installing dependencies =="
  uv venv .venv --python 3.11
  uv pip install -p "$PYTHON" -r requirements.txt
fi

echo "== M1: preparing C-MAPSS data =="
"$PYTHON" scripts/prepare_data.py \
  --subset FD001 \
  --out data/processed/cmapss \
  --T 24

echo "== M2/M3: full training (iter_pre=5000, iter_main=10000) on device=$DEVICE =="
"$PYTHON" scripts/train.py \
  --config "$CONFIG" \
  --out "$CKPT_DIR" \
  --device "$DEVICE"

echo "== M4: recursive PC sampling (n_samples=$N_SAMPLES, n_steps=$N_STEPS) =="
"$PYTHON" scripts/sample.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT_DIR/ckpt_latest.pt" \
  --n-samples "$N_SAMPLES" \
  --n-steps "$N_STEPS" \
  --out outputs/samples/cmapss.npy \
  --device "$DEVICE"

echo "== M5: evaluation (n_seeds=$N_SEEDS) =="
"$PYTHON" scripts/evaluate.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT_DIR/ckpt_latest.pt" \
  --n-seeds "$N_SEEDS" \
  --device "$DEVICE"

echo "== done. reports in outputs/reports/ =="
