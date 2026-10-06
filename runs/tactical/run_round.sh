#!/bin/bash
# usage: run_round.sh <spec-name> ; 3 shards, each retried (resume) up to 3 times
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
for i in 0 1 2; do
  ( for a in 1 2 3; do $PY -m neural_trade.cli screen configs/tactical/$1.yaml --store runs/tactical --shard $i/3 >> runs/tactical/$1_$i.log 2>&1 && break; done ) &
done
wait
