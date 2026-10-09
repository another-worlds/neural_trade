#!/bin/bash
# two processes at a time on the GPU (2000 MB each = 4000 MB total), below-normal priority set inside pretrain2.py
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_wt_jepa2/src
PY=C:/Users/Step/miniforge3/envs/nt/python
run() { $PY pretrain2.py --variant $1 --steps 30000 > ckpt/$1/pretrain.out 2>&1; }
run c & run ctl & wait
run a & run b & wait
