#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -f $R/chain5.done ]; do sleep 20; done
J=""
for v in mRich mPerm pmPerm pmRich pmG5; do J="$J $R/meta.py run --bars 1min --feat tb7 --variant $v ;; $R/meta.py run --bars 1h --feat tb7 --variant $v ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/D -- $J > $R/logs_D.out 2>&1
echo done > $R/chain6.done
