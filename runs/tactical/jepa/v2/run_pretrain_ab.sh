#!/bin/bash
# a and b started while c (run_pretrain.sh, 2000 MB) was still training: 1000 MB each, 4000 MB in total
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_wt_jepa2/src
PY=C:/Users/Step/miniforge3/envs/nt/python
run() { $PY pretrain2.py --variant $1 --steps 30000 --mem 1000 > ckpt/$1/pretrain.out 2>&1; }
run a & run b & wait
