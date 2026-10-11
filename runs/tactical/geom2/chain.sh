#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ $(grep -c "rc " $R/logs_ro.out) -lt 6 ]; do sleep 20; done
# stage A: vol (minute 6 shards + hourly) and the minute meta build
$PY $R/launch.py 6 $R/logs/A -- $R/volgeom.py minute --shard 0 --of 6 ";;" $R/volgeom.py minute --shard 1 --of 6 ";;" $R/volgeom.py minute --shard 2 --of 6 ";;" \
  $R/volgeom.py minute --shard 3 --of 6 ";;" $R/volgeom.py minute --shard 4 --of 6 ";;" $R/volgeom.py minute --shard 5 --of 6 ";;" \
  $R/volgeom.py hourly ";;" $R/meta.py build --bars 1min > $R/logs_A.out 2>&1
# stage B: minute meta runs (long) first, then hourly meta runs, then hourly direction
J=""
for v in base mG5 mGR pmGR; do J="$J $R/meta.py run --bars 1min --feat tb7 --variant $v ;;"; done
for f in tb7 rich; do for v in base mG5 mGR pmGR; do J="$J $R/meta.py run --bars 1h --feat $f --variant $v ;;"; done; done
for s in 0 1 2 3 4 5; do J="$J $R/hdir.py run --shard $s --of 6 ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/B -- $J > $R/logs_B.out 2>&1
echo done > $R/chain.done
