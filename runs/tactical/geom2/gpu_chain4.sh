#!/bin/bash
# Step 6d: the patch network with and without geometry on the 3-year span, seed 0, 24 slices (stops at 07:15 if not finished).
cd /d/nt/nt_tactical_geom2/runs/tactical/newnet
export PYTHONPATH=/d/nt/nt_tactical_geom2/src PYTHONIOENCODING=utf-8 SERIES_CACHE=/d/nt/nt_tactical_newnet/runs/tactical/newnet/cache OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
R=../geom2; G=/d/nt/nt_tactical_geom2/runs/tactical/newnet/cache/series_geom.npy; O=/d/nt/nt_tactical_geom2/runs/tactical/geom2/newnet_results.jsonl
$PY train.py --slices 24 --span 3y --arch patch --train-lin --tag geom2_patch3y_base --out $O > $R/logs/gpu_patch3y_base.log 2>&1
[ -f $R/gpu_stop ] && exit
$PY train.py --slices 24 --span 3y --arch patch --train-lin --geom $G --tag geom2_patch3y_geom --out $O > $R/logs/gpu_patch3y_geom.log 2>&1
echo done > $R/gpu_chain4.done
