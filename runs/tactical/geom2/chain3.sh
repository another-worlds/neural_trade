#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -s $R/logs_A.out ] || [ $(grep -c "rc " $R/logs_A.out) -lt 8 ]; do sleep 15; done
J=""
for v in base mG5 mGR pmGR; do J="$J $R/meta.py run --bars 1min --feat tb7 --variant $v ;;"; done
for s in 0 1 2 3 4 5; do J="$J $R/volgeom.py minutevec --shard $s --of 6 ;;"; done
for s in 0 1 2 3 4 5; do J="$J $R/readout.py --shard $s --of 6 --models logreg_c01,logreg_c001,hgb_reg,mlp_reg ;;"; done
for f in tb7 rich; do for v in base mG5 mGR pmGR; do J="$J $R/meta.py run --bars 1h --feat $f --variant $v ;;"; done; done
for s in 0 1 2 3 4 5; do J="$J $R/hdir.py run --shard $s --of 6 ;;"; done
J="$J $R/featscreen.py"
$PY $R/launch.py 6 $R/logs/B -- $J > $R/logs_B.out 2>&1
echo done > $R/chain3.done
