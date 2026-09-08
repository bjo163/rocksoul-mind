#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/trainer"

DATA_PATH="${DATA_PATH:-../dataset/rocksoul_iden/pretrain_iden_train.jsonl}"
SAVE_WEIGHT="${SAVE_WEIGHT:-rocksoul_iden_pretrain}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-384}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"

python train_pretrain.py \
  --data_path "$DATA_PATH" \
  --from_weight pretrain \
  --save_weight "$SAVE_WEIGHT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --accumulation_steps "$ACCUMULATION_STEPS" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --learning_rate "$LEARNING_RATE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --save_interval 500 \
  --log_interval 50 \
  --seed 163
