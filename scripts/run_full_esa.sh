#!/usr/bin/env bash
# Full-scale TSGM ESA run: data prep -> train -> recursive PC sampling (n_steps=1000)
# -> 10-seed discriminative/predictive/t-SNE evaluation. Iteration counts, score-net type,
# SDE, etc. all come from configs/esa.yaml. See docs/reproduction_plan.md.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=".venv/bin/python"
CONFIG="configs/esa.yaml"
CKPT_DIR="outputs/checkpoints/esa_full"
DEVICE="${DEVICE:-cpu}"
N_SAMPLES="${N_SAMPLES:-100}"
N_STEPS="${N_STEPS:-1000}"
N_SEEDS="${N_SEEDS:-10}"

if [ ! -x "$PYTHON" ]; then
  echo "== creating venv and installing dependencies =="
  uv venv .venv --python 3.11
  uv pip install -p "$PYTHON" -r requirements.txt
fi

echo "== M7: preparing ESA (Mission1) data =="
"$PYTHON" scripts/prepare_data_esa.py \
  --out data/processed/esa \
  --T 24

echo "== training (counts from $CONFIG) on device=$DEVICE =="
"$PYTHON" scripts/train.py \
  --config "$CONFIG" \
  --out "$CKPT_DIR" \
  --device "$DEVICE"

if [ -f "$CKPT_DIR/ckpt_best.pt" ]; then CKPT="$CKPT_DIR/ckpt_best.pt"; else CKPT="$CKPT_DIR/ckpt_latest.pt"; fi
echo "== using checkpoint $CKPT =="

echo "== recursive PC sampling (n_samples=$N_SAMPLES, n_steps=$N_STEPS) =="
"$PYTHON" scripts/sample.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --n-samples "$N_SAMPLES" \
  --n-steps "$N_STEPS" \
  --out outputs/samples/esa.npy \
  --device "$DEVICE"

echo "== evaluation (n_seeds=$N_SEEDS) =="
"$PYTHON" scripts/evaluate.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --n-seeds "$N_SEEDS" \
  --device "$DEVICE"

echo "== done. reports in outputs/reports/ =="
