#!/bin/bash
# variant c, 120k steps, seeds 0 and 1, two processes on the GPU (2000 MB each), below-normal priority set inside pretrain3.py
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_tactical_jepa3/src
PY=C:/Users/Step/miniforge3/envs/nt/python
for s in 0 1; do mkdir -p ckpt/c_s$s; $PY pretrain3.py --variant c --steps 120000 --seed $s --mem 2000 --out ckpt/c_s$s > ckpt/c_s$s/pretrain.out 2>&1 & done
wait
