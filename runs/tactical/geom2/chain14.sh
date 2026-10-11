#!/bin/bash
cd /d/nt/nt_tactical_geom2
export PYTHONPATH=$PWD/src
PY=C:/Users/Step/miniforge3/envs/nt/python
R=runs/tactical/geom2
J=""
for s in 0 1 2; do J="$J $R/readout.py --shard $s --of 3 --sets gc3,gc5,gc10,gc20,tb7+gc3,tb7+gc5,tb7+gc10,tb7+gc20 --models logreg,logreg_c001,hgb_reg ;;"; done
for s in 0 1 2; do J="$J $R/longdir.py --shard $s --of 3 --sets gc3,gc5,gc10,gc20,tb7+gc3,tb7+gc5,tb7+gc10,tb7+gc20 --models logreg,logreg_c001 ;;"; done
J=${J% ;;}
$PY $R/launch.py 6 $R/logs/M -- $J > $R/logs_M.out 2>&1
echo done > $R/chain14.done
