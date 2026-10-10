#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=D:/nt/nt_tactical_jepa4/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
mkdir -p ckpt/e1L; $PY pretrain4.py --name e1L --variant c --span-end "2021-12-31 23:59" --steps 34000 > ckpt/e1L/pretrain.out 2>&1
