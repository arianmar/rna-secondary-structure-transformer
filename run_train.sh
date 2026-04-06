#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-new}"

EPOCHS=20
SNAP_EVERY=5
OUTPUT_DIR="models"

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ "$MODE" == "new" ]]; then
  if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "usage: ./run_train.sh new /pfad/zum/dataset.pkl [epochs] [snap_every] [output_dir]" >&2
    exit 1
  fi

  DATA_PATH="$2"

  if [[ $# -ge 3 ]]; then
    if ! is_int "$3"; then
      echo "epochs must be an integer" >&2
      exit 1
    fi
    EPOCHS="$3"
  fi

  if [[ $# -ge 4 ]]; then
    if ! is_int "$4"; then
      echo "snap_every must be an integer" >&2
      exit 1
    fi
    SNAP_EVERY="$4"
  fi

  if [[ $# -eq 5 ]]; then
    OUTPUT_DIR="$5"
  fi

  python3 train.py \
    --mode new \
    --data-path "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --snap-every "$SNAP_EVERY" \
    --batch-train 64 \
    --batch-val 64 \
    --epochs "$EPOCHS" \
    --patience 3 \
    --min-delta 0.0 \
    --lr 5e-4 \
    --weight-decay 1e-2 \
    --eta-min 1e-5 \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 4 \
    --d-ff 1024 \
    --dropout 0.1 \
    --log-every 250 \
    --split-seed 1337

elif [[ "$MODE" == "current" ]]; then
  if [[ $# -gt 5 ]]; then
    echo "usage: ./run_train.sh current [checkpoint_dir_or_pt] [epochs] [snap_every] [output_dir]" >&2
    exit 1
  fi

  DATA_PATH=""

  if [[ $# -ge 2 ]]; then
    if is_int "$2"; then
      EPOCHS="$2"

      if [[ $# -ge 3 ]]; then
        if ! is_int "$3"; then
          echo "snap_every must be an integer" >&2
          exit 1
        fi
        SNAP_EVERY="$3"
      fi

      if [[ $# -ge 4 ]]; then
        OUTPUT_DIR="$4"
      fi

    else
      DATA_PATH="$2"

      if [[ $# -ge 3 ]]; then
        if ! is_int "$3"; then
          echo "epochs must be an integer" >&2
          exit 1
        fi
        EPOCHS="$3"
      fi

      if [[ $# -ge 4 ]]; then
        if ! is_int "$4"; then
          echo "snap_every must be an integer" >&2
          exit 1
        fi
        SNAP_EVERY="$4"
      fi

      if [[ $# -eq 5 ]]; then
        OUTPUT_DIR="$5"
      fi
    fi
  fi

  if [[ -n "$DATA_PATH" ]]; then
    python3 train.py \
      --mode current \
      --data-path "$DATA_PATH" \
      --output "$OUTPUT_DIR" \
      --snap-every "$SNAP_EVERY" \
      --batch-train 64 \
      --batch-val 64 \
      --epochs "$EPOCHS" \
      --patience 3 \
      --min-delta 0.0 \
      --lr 5e-4 \
      --weight-decay 1e-2 \
      --eta-min 1e-5 \
      --d-model 256 \
      --n-heads 8 \
      --n-layers 4 \
      --d-ff 1024 \
      --dropout 0.1 \
      --log-every 250 \
      --split-seed 1337
  else
    python3 train.py \
      --mode current \
      --output "$OUTPUT_DIR" \
      --snap-every "$SNAP_EVERY" \
      --batch-train 64 \
      --batch-val 64 \
      --epochs "$EPOCHS" \
      --patience 3 \
      --min-delta 0.0 \
      --lr 5e-4 \
      --weight-decay 1e-2 \
      --eta-min 1e-5 \
      --d-model 256 \
      --n-heads 8 \
      --n-layers 4 \
      --d-ff 1024 \
      --dropout 0.1 \
      --log-every 250 \
      --split-seed 1337
  fi

else
  echo "mode must be 'new' or 'current'" >&2
  exit 1
fi