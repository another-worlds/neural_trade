#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
J=""
for s in 0 1 2 3 4 5; do J="$J $R/longdir.py --shard $s --of 6 --span 1576800 --sets tb7,rich,gR,tb7+gR,rich+gR --models logreg,logreg_c001,hgb_reg ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/N -- $J > $R/logs_N.out 2>&1
echo done > $R/chain15.done
