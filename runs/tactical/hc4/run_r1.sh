#!/bin/bash
# Round 1: the base's missing run plus 6 variants x 3 shards; 3 processes at a time; each task retried up to 3 times.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8 PY=C:/Users/Step/miniforge3/envs/nt/python
one() { set -- $1; n=$1; sh=$2; tag=${sh//\//of}; for a in 1 2 3; do $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical --shard $sh >> runs/tactical/hc4/${n}_${tag}.log 2>&1 && break; done; echo done $n $sh; }
export -f one
{ echo "cand_c2_base 0/1"; for v in calval calgrad ep12 look120 nophys bs1024; do for i in 0 1 2; do echo "hc4_$v $i/3"; done; done; } | xargs -P 3 -I{} bash -c 'one "{}"'
echo done hc4 r1
