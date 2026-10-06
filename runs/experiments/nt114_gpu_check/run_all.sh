#!/bin/bash
# NT-114 GPU check. Run from D:/nt/nt_wt_114. One GPU job at a time.
PY=C:/Users/Step/miniforge3/envs/nt/python
export PYTHONPATH=D:/nt/nt_wt_114/src PYTHONIOENCODING=utf-8
D=runs/experiments/nt114_gpu_check
for spec in "det_gru_r1 true" "det_gru_r2 true" "det_gru_r3 true" "det_cudnn_ref false"; do
  set -- $spec
  nvidia-smi dmon -s um -c 5 > $D/$1.gpucheck.txt
  $PY $D/harness_train_once.py --config configs/default.yaml --fold-index -3 --epochs 3 --seed 777 \
     --deterministic 1 --override DETERMINISTIC_GRU=$2 --runs-dir $D/runs --name $1 \
     > $D/$1.stdout 2> $D/$1.stderr
  echo "$1 exit $?" >> $D/progress.txt
done
echo ALLDONE >> $D/progress.txt
