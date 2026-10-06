#!/bin/bash
# usage: run_hc2.sh <variant> ; runs the 5 climb chunk specs of a variant one after another, 3 shards each,
# each shard retried up to 3 times (resume is idempotent).
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
for c in c0 c1 c2 c3 c4; do
  n=hc2_$1_$c
  for i in 0 1 2; do
    ( for a in 1 2 3; do $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical --shard $i/3 >> runs/tactical/${n}_$i.log 2>&1 && break; done ) &
  done
  wait
done
echo done $1
