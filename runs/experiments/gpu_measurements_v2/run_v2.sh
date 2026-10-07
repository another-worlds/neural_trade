#!/bin/bash
# NT-173: N = 1,2,3 (x3 repeats) and N = 4 only if N = 3 leaves 2 GB headroom. GPU-free check before every batch.
cd /d/nt/nt_wt_173
export PYTHONPATH=/d/nt/nt_wt_173/src PYTHONIOENCODING=utf-8
PY="C:/Users/Step/miniforge3/envs/nt/python"
E=runs/experiments/gpu_measurements_v2
OUT=$E/throughput
LOG=$E/gpu_free_checks.log
mkdir -p $OUT
gpu_free() {
  local s; s=$(nvidia-smi dmon -s um -c 10 | awk '$1 ~ /^[0-9]+$/ {print $2, $8}')
  echo "$(date -u +%FT%TZ) check: $(echo "$s" | tr '\n' ';')" >> "$LOG"
  echo "$s" | $PY -c "
import sys,statistics as st
r=[l.split() for l in sys.stdin if l.strip()]
sm=[float(a) for a,b in r]; fb=[float(b) for a,b in r]
ok=max(fb)<=2000
print('median_sm',st.median(sm),'max_fb',max(fb),'FREE' if ok else 'BUSY (judged by fb; desktop sm baseline ~40)')
sys.exit(0 if ok else 1)" >> "$LOG"
}
wait_free() {
  local deadline=$(( $(date +%s) + 3000 ))
  until gpu_free; do
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "GPU_BUSY_TIMEOUT" >> "$LOG"; exit 3; fi
    sleep 60
  done
}
run_batch() {
  wait_free
  echo "=== N=$1 repeat=$2 $(date -u +%FT%TZ) ===" >> "$LOG"
  $PY $E/harness_concurrent.py --n "$1" --repeat "$2" --seed-base "$3" --epochs 3 --fold-index -3 --out "$OUT" >> $E/batches.log 2>&1
}
for r in 1 2 3; do run_batch 1 $r $((9000+r)); done
for r in 1 2 3; do run_batch 2 $r $((9100+10*r)); done
for r in 1 2 3; do run_batch 3 $r $((9200+10*r)); done
echo "PART_DONE_THROUGH_N3 $(date -u +%FT%TZ)" >> "$LOG"
