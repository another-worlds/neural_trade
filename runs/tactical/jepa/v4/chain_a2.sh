#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_tactical_jepa4/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
run() { n=$1; shift; mkdir -p ckpt/$n; $PY pretrain4.py --name $n "$@" > ckpt/$n/pretrain.out 2>&1; }
while kill -0 7072 2>/dev/null; do sleep 20; done
run e4 --variant c --ctx 120 --steps 30000
