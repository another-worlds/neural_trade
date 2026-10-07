#!/bin/bash
# Waits for the candidate check to finish, then runs the three arms one after another and records wall times.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
until grep -q "^done cand all" runs/tactical/cand_0805/run.out 2>/dev/null; do sleep 30; done
T=runs/tactical/bench_1d/times.txt; : > $T
arm() { echo "$1 start $(date +%s)" >> $T; }
fin() { echo "$1 end $(date +%s)" >> $T; }
arm bs256_x1;  $PY -m neural_trade.cli screen configs/tactical/bench_bs256_x1.yaml --store runs/tactical > runs/tactical/bench_1d/bs256_x1.log 2>&1; fin bs256_x1
arm bs256_x3
for i in 0 1 2; do $PY -m neural_trade.cli screen configs/tactical/bench_bs256_x3.yaml --store runs/tactical --shard $i/3 > runs/tactical/bench_1d/bs256_x3_$i.log 2>&1 & done
wait; fin bs256_x3
arm bs1024_x1; $PY -m neural_trade.cli screen configs/tactical/bench_bs1024_x1.yaml --store runs/tactical > runs/tactical/bench_1d/bs1024_x1.log 2>&1; fin bs1024_x1
echo done bench
