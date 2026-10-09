#!/bin/bash
# usage: run_arm.sh <logname> <train.py args...>   (newnet2: one arm in one process; below-normal priority and the 3000 MB GPU cap are in train.py)
cd "$(dirname "$0")"; mkdir -p logs
PY=C:/Users/Step/miniforge3/envs/nt/python
export PYTHONIOENCODING=utf-8 PYTHONPATH=/d/nt/nt_tactical_newnet2/src
name=$1; shift
$PY train.py "$@" > logs/$name.log 2>&1
