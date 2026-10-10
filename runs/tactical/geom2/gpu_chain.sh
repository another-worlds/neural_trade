#!/bin/bash
# Step 6: geometry in the new network's last-bar vector, 24 slices x 1y, 1 seed; linear arch and patch arch with --train-lin, each with and without.
cd /d/nt/nt_tactical_geom2/runs/tactical/newnet
export PYTHONPATH=/d/nt/nt_tactical_geom2/src PYTHONIOENCODING=utf-8 SERIES_CACHE=/d/nt/nt_tactical_newnet/runs/tactical/newnet/cache OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
R=../geom2; G=/d/nt/nt_tactical_geom2/runs/tactical/newnet/cache/series_geom.npy

$PY build_geom.py --check > $R/logs/geom_check.log 2>&1
for arm in "linear base" "linear geom" "patch base" "patch geom"; do
  set -- $arm; a=$1; g=$2; extra=""; [ "$g" = geom ] && extra="--geom $G"
  [ -f $R/gpu_stop ] && break
  $PY train.py --slices 24 --span 1y --arch $a --train-lin $extra --tag geom2_${a}_${g} --out /d/nt/nt_tactical_geom2/runs/tactical/geom2/newnet_results.jsonl > $R/logs/gpu_${a}_${g}.log 2>&1
done
echo done > $R/gpu_chain.done
