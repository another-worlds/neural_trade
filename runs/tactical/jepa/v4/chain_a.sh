#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_tactical_jepa4/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
run() { n=$1; shift; mkdir -p ckpt/$n; $PY pretrain4.py --name $n "$@" > ckpt/$n/pretrain.out 2>&1; }
run e1 --variant c --span-end "2021-12-31 23:59" --steps 60000
run e3 --variant c --d 64 --layers 3 --heads 4 --ffn 128 --steps 60000
