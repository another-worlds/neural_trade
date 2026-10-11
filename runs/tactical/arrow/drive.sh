#!/bin/bash
# usage: drive.sh PART OF  -- repeats ~100 s chunks of run.py until its part is done (each chunk = a separate process, D-063)
PY=C:/Users/Step/miniforge3/envs/nt/python
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_wt_arrow/src PYTHONIOENCODING=utf-8 TF_FORCE_GPU_ALLOW_GROWTH=true
while true; do
  $PY run.py --part "$1" --of "$2" --budget 100 >> "log_part$1.txt" 2>&1
  rc=$?
  if [ $rc -ne 3 ]; then echo "part $1 finished rc=$rc $(date)" >> "log_part$1.txt"; break; fi
done
