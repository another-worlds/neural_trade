#!/bin/bash
cd /d/nt_exp_gpu_measurements_v1
export PYTHONPATH=/d/nt_exp_gpu_measurements_v1/src
export PYTHONIOENCODING=utf-8
PY="C:/Users/Step/miniforge3/envs/nt/python"
E=runs/experiments/gpu_measurements_v1
OUT=$E/determinism
LOG=$E/part_b.log
mkdir -p $OUT
: > "$LOG"
gpu_free() {
  # RUNBOOK GPU-free check: 10 one-second samples; busy if median sm > 30 or max fb > 2000 MB
  local s; s=$(nvidia-smi dmon -s um -c 10 | awk '$1 ~ /^[0-9]+$/ {print $2, $8}')
  echo "$(date -u +%FT%TZ) check: $(echo "$s" | tr '\n' ';')" >> "$LOG"
  echo "$s" | $PY -c "
import sys,statistics as st
r=[l.split() for l in sys.stdin if l.strip()]
sm=[float(a) for a,b in r]; fb=[float(b) for a,b in r]
ok=st.median(sm)<=30 and max(fb)<=2000
print('median_sm',st.median(sm),'max_fb',max(fb),'FREE' if ok else 'BUSY')
sys.exit(0 if ok else 1)" >> "$LOG"
}
wait_free() {
  local deadline=$(( $(date +%s) + 3000 ))
  until gpu_free; do
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "GPU_BUSY_TIMEOUT" >> "$LOG"; exit 3; fi
    sleep 60
  done
}
# record the determinism state each arm actually gets (CPU probe, no training)
for v in unset 0; do
  if [ $v = unset ]; then envp=""; else envp="TF_DETERMINISTIC_OPS=0"; fi
  env $envp CUDA_VISIBLE_DEVICES=-1 $PY -c "
import neural_trade, os, tensorflow as tf
from tensorflow.python.util import _pywrap_determinism as d
print('probe arm_env=$v TF_DETERMINISTIC_OPS=',os.environ.get('TF_DETERMINISTIC_OPS'),'is_enabled=',d.is_enabled(),'tf32=',tf.config.experimental.tensor_float_32_execution_enabled())
tf.config.experimental.enable_op_determinism(); print('probe after enable_op_determinism is_enabled=',d.is_enabled())" 2>/dev/null >> "$LOG"
done
wait_free
for r in 1 2 3; do
  for arm in on off; do
    echo "=== det_$arm r$r $(date -u +%FT%TZ) ===" >> "$LOG"
    if [ $arm = on ]; then
      $PY $E/harness_train_once.py --fold-index -3 --epochs 3 --seed 777 --deterministic 1 --runs-dir $OUT --name det_on_r$r > $OUT/det_on_r$r.stdout 2> $OUT/det_on_r$r.stderr
    else
      TF_DETERMINISTIC_OPS=0 $PY $E/harness_train_once.py --fold-index -3 --epochs 3 --seed 777 --deterministic 0 --runs-dir $OUT --name det_off_r$r > $OUT/det_off_r$r.stdout 2> $OUT/det_off_r$r.stderr
    fi
    echo "rc=$? $(tail -1 $OUT/det_${arm}_r$r.stdout)" >> "$LOG"
  done
done
echo "PART_B_DONE $(date -u +%FT%TZ)" >> "$LOG"
