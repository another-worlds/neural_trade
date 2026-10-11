#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -f $R/chain8.done ]; do sleep 20; done
J=""
for v in base pmGR pmRich pmG5; do J="$J $R/meta.py run --bars 1min --feat tb7 --variant $v --primary hgb ;; $R/meta.py run --bars 1h --feat tb7 --variant $v --primary hgb ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/G -- $J > $R/logs_G.out 2>&1
echo done > $R/chain9.done
