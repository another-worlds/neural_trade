#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_tactical_jepa4/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
run() { n=$1; shift; mkdir -p ckpt/$n; $PY pretrain4.py --name $n "$@" > ckpt/$n/pretrain.out 2>&1; }
while kill -0 6356 2>/dev/null; do sleep 20; done
run m30 --variant c --mask-ratio 0.3 --steps 30000
run m70 --variant c --mask-ratio 0.7 --steps 30000
run e3 --variant c --d 64 --layers 3 --heads 4 --ffn 128 --steps 30000
