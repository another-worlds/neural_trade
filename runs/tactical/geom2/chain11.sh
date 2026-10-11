#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -f $R/chain10.done ]; do sleep 20; done
$PY $R/meta.py build --bars 1min_a > $R/logs/build_a.log 2>&1
J=""
for v in base mG5 mGR pmGR pmG5 pmRich pmPerm mRich mPerm; do J="$J $R/meta.py run --bars 1min_a --feat tb7 --variant $v ;; $R/meta.py run --bars 1min_a --feat rich --variant $v ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/I -- $J > $R/logs_I.out 2>&1
echo done > $R/chain11.done
