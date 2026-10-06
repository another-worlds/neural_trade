#!/bin/bash
# usage: run_chunks.sh <variant> <chunk>... ; 3 shards per chunk, each shard retried (resume) up to 3 times.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
v=$1; shift
for c in "$@"; do
  n=hc2_${v}_$c
  for i in 0 1 2; do
    ( for a in 1 2 3; do $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical --shard $i/3 >> runs/tactical/${n}_$i.log 2>&1 && break; done ) &
  done
  wait
done
echo done $v "$@"
