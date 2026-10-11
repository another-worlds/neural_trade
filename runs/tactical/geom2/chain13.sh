#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
$PY $R/launch.py 2 $R/logs/K -- $R/meta.py build --bars 5min ";;" $R/meta.py build --bars 15min > $R/logs_K.out 2>&1
J=""
for b in 5min 15min; do
  for v in base mG5 mGR pmGR pmG5 pmRich pmPerm mRich mPerm; do J="$J $R/meta.py run --bars $b --feat tb7 --variant $v ;;"; done
  for v in base pmGR pmRich; do J="$J $R/meta.py run --bars $b --feat tb7 --variant $v --primary hgb ;;"; done
done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/L -- $J > $R/logs_L.out 2>&1
echo done > $R/chain13.done
