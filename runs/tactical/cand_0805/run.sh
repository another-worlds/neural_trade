#!/bin/bash
# Candidate check (SPEC.md): C1 (2 screen specs) and C2 (2 seven-day specs, 2 shards each); 3 processes at a time.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8 PY=C:/Users/Step/miniforge3/envs/nt/python
one() { set -- $1; n=$1; sh=$2; arg=""; [ -n "$sh" ] && arg="--shard $sh"; tag=${sh//\//of};
  for a in 1 2 3; do $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical $arg >> runs/tactical/${n}_${tag}.log 2>&1 && break; done; echo done $n $sh; }
export -f one
printf "cand_c1_skip\ncand_c1_base\ncand_c2_skip 0/2\ncand_c2_skip 1/2\ncand_c2_base 0/2\ncand_c2_base 1/2\n" | xargs -P 3 -I{} bash -c 'one "{}"'
echo done cand all
