#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
J=""
for v in base pmGR pmRich pmG5; do J="$J $R/meta.py run --bars 1min_a --feat tb7 --variant $v --primary hgb ;;"; done
J=${J% ;;}
$PY $R/launch.py 4 $R/logs/J -- $J > $R/logs_J.out 2>&1
echo done > $R/chain12.done
