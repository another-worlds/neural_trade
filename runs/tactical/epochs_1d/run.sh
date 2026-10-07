#!/bin/bash
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
until grep -q "^done bench" runs/tactical/bench_1d/run.out 2>/dev/null; do sleep 20; done
for n in ep1d_bs256_e40 ep1d_bs64_e14; do
  for i in 0 1; do $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical --shard $i/2 > runs/tactical/epochs_1d/${n}_$i.log 2>&1 & done
  wait
done
echo done epochs
