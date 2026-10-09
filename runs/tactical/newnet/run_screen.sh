#!/bin/bash
# newnet2 step 1: screen of the three early-stopping fixes on 6 slices x 1 y (lane 1: train-lin, then warm-up; lane 2: small patch)
cd "$(dirname "$0")"; mkdir -p logs
PY=C:/Users/Step/miniforge3/envs/nt/python
export PYTHONIOENCODING=utf-8 PYTHONPATH=/d/nt/nt_tactical_newnet2/src
lane=$1
if [ "$lane" = 1 ]; then
  $PY train.py --slices 6 --span 1y --arch patch --train-lin --tag screen_trainlin > logs/screen_trainlin.log 2>&1
  $PY train.py --slices 6 --span 1y --arch patch --warm 2 --tag screen_warm2 > logs/screen_warm2.log 2>&1
else
  $PY train.py --slices 6 --span 1y --arch patchS --tag screen_patchS > logs/screen_patchS.log 2>&1
fi
