#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
while [ ! -f $R/chain7.done ]; do sleep 20; done
J=""
for s in 0 1 2 3 4 5; do J="$J $R/longdir.py --shard $s --of 6 --models hgb,mlp,mlp_reg --sets tb7,rich,g5,gR,tb7+gR,rich+gR ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/F -- $J > $R/logs_F.out 2>&1
echo done > $R/chain8.done
