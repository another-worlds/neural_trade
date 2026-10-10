#!/bin/bash
# Step 6b: seeds 1 and 2 of the two patch arms (seed 0 is in newnet_results.jsonl), after gpu_chain.sh.
cd /d/nt/nt_tactical_geom2/runs/tactical/newnet
export PYTHONPATH=/d/nt/nt_tactical_geom2/src PYTHONIOENCODING=utf-8 SERIES_CACHE=/d/nt/nt_tactical_newnet/runs/tactical/newnet/cache OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
R=../geom2; G=/d/nt/nt_tactical_geom2/runs/tactical/newnet/cache/series_geom.npy; O=/d/nt/nt_tactical_geom2/runs/tactical/geom2/newnet_results.jsonl
while [ ! -f $R/gpu_chain.done ]; do sleep 30; done
for s in 1 2; do
  $PY train.py --slices 24 --span 1y --arch patch --train-lin --seed0 $s --tag geom2_patch_base_s$s --out $O > $R/logs/gpu_patch_base_s$s.log 2>&1
  [ -f $R/gpu_stop ] && break
  $PY train.py --slices 24 --span 1y --arch patch --train-lin --geom $G --seed0 $s --tag geom2_patch_geom_s$s --out $O > $R/logs/gpu_patch_geom_s$s.log 2>&1
  [ -f $R/gpu_stop ] && break
done
echo done > $R/gpu_chain2.done
