#!/bin/bash
# 11 specs (default + 10 candidates), 3 processes at a time; each spec retried up to 3 times (resume is idempotent).
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
one() { n=$1; for a in 1 2 3; do $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical >> runs/tactical/$n.log 2>&1 && break; done; echo done $n; }
export -f one; export PY
printf "hc3_default\nhc3_cand01\nhc3_cand02\nhc3_cand03\nhc3_cand04\nhc3_cand05\nhc3_cand06\nhc3_cand07\nhc3_cand08\nhc3_cand09\nhc3_cand10\n" | xargs -P 3 -I{} bash -c 'one {}'
echo done hc3 all
