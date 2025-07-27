#!/usr/bin/env bash

# list of models
models=(
  emp_small_wide
  emp_small_thin
  emp_tiny_wide
  empt_tiny_thin
)

# list of rkd_on values
rkd_values=(
  x_agent
  null
)

# base command arguments
DATA_ROOT="/Users/kbf/Desktop/Kurse/DL Lab/DL_lab_project/data/av2/"
BATCH_SIZE=48

for model in "${models[@]}"; do
  for rkd in "${rkd_values[@]}"; do
    echo "=== Running model=${model}, training.rkd.rkd_on=${rkd} ==="
    python train_rkd.py \
      data_root="${DATA_ROOT}" \
      batch_size=${BATCH_SIZE} \
      model="${model}" \
      training.rkd.rkd_on="${rkd}"
  done
done
