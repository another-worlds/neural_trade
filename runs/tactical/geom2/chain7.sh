#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -f $R/chain6.done ]; do sleep 20; done
J=""
for v in base mG5 mGR pmGR pmPerm; do
  for cfg in "30 1.0" "60 1.0" "15 1.5" "15 0.7"; do set -- $cfg; J="$J $R/meta.py run --bars 1min --feat tb7 --variant $v --T $1 --M $2 ;;"; done
  for cfg in "24 1.5" "12 1.0" "12 2.0" "48 1.5"; do set -- $cfg; J="$J $R/meta.py run --bars 1h --feat tb7 --variant $v --T $1 --M $2 ;;"; done
done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/E -- $J > $R/logs_E.out 2>&1
echo done > $R/chain7.done
