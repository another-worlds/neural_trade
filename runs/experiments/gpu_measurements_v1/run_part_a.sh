#!/bin/bash
set -e
cd /d/nt_exp_gpu_measurements_v1
export PYTHONPATH=/d/nt_exp_gpu_measurements_v1/src
export PYTHONIOENCODING=utf-8
PY="C:/Users/Step/miniforge3/envs/nt/python"
OUT=runs/experiments/gpu_measurements_v1/throughput
LOG=runs/experiments/gpu_measurements_v1/part_a.log
: > "$LOG"

run_batch() {
  local n=$1 rep=$2 base=$3
  echo "=== N=$n repeat=$rep seed_base=$base $(date -u +%FT%TZ) ===" >> "$LOG"
  $PY runs/experiments/gpu_measurements_v1/harness_concurrent.py --n "$n" --repeat "$rep" \
      --seed-base "$base" --epochs 3 --fold-index -3 --out "$OUT" >> "$LOG" 2>&1
}

run_batch 1 1 9001
run_batch 1 2 9002
run_batch 2 0 9100
run_batch 2 1 9110
run_batch 2 2 9120
run_batch 3 0 9200
run_batch 3 1 9210
run_batch 3 2 9220
run_batch 4 0 9300
run_batch 4 1 9310
run_batch 4 2 9320

echo "PART_A_DONE $(date -u +%FT%TZ)" >> "$LOG"
