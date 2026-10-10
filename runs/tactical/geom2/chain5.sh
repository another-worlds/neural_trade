#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -f $R/chain3.done ]; do sleep 20; done
J=""
for s in 0 1 2 3 4 5; do J="$J $R/longdir.py --shard $s --of 6 ;;"; done
for v in base mG5 mGR pmGR; do
  J="$J $R/meta.py run --bars 1min --feat rich --variant $v ;; $R/meta.py run --bars 1min --feat tb7 --variant $v --q 0.05 ;; $R/meta.py run --bars 1min --feat tb7 --variant $v --q 0.2 ;;"
  J="$J $R/meta.py run --bars 1h --feat tb7 --variant $v --q 0.1 ;; $R/meta.py run --bars 1h --feat tb7 --variant $v --q 0.3 ;;"
done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/C -- $J > $R/logs_C.out 2>&1
echo done > $R/chain5.done
