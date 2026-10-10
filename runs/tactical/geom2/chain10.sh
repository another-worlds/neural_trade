#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -f $R/chain9.done ]; do sleep 20; done
J=""
for v in trend band osc vol slope pos cross squeeze; do J="$J $R/meta.py run --bars 1min --feat tb7 --variant pmG_$v ;;"; done
for v in trend band osc vol slope pos cross squeeze; do J="$J $R/meta.py run --bars 1h --feat tb7 --variant pmG_$v ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/H -- $J > $R/logs_H.out 2>&1
echo done > $R/chain10.done
