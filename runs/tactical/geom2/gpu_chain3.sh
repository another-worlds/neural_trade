#!/bin/bash
# Step 6c: the regression with and without geometry on the 3-YEAR span (more data may suit 164 extra features), seed 0, 24 slices.
cd /d/nt/nt_tactical_geom2/runs/tactical/newnet
export PYTHONPATH=/d/nt/nt_tactical_geom2/src PYTHONIOENCODING=utf-8 SERIES_CACHE=/d/nt/nt_tactical_newnet/runs/tactical/newnet/cache OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
R=../geom2; G=/d/nt/nt_tactical_geom2/runs/tactical/newnet/cache/series_geom.npy; O=/d/nt/nt_tactical_geom2/runs/tactical/geom2/newnet_results.jsonl
$PY train.py --slices 24 --span 3y --arch linear --tag geom2_linear3y_base --out $O > $R/logs/gpu_linear3y_base.log 2>&1
[ -f $R/gpu_stop ] && exit
$PY train.py --slices 24 --span 3y --arch linear --geom $G --tag geom2_linear3y_geom --out $O > $R/logs/gpu_linear3y_geom.log 2>&1
echo done > $R/gpu_chain3.done
