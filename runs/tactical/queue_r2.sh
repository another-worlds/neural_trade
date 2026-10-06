#!/bin/bash
# Round 2 queue: each variant over the 40 climb slices x 3 seeds, run after the baseline finishes.
# Variants fixed before any result (see make_hc2.py rule). One change per variant, Config-only (hyp1 switches exist).
cd /d/nt/nt_tactical
PY=C:/Users/Step/miniforge3/envs/nt/python
until grep -q "^done base" runs/tactical/run_hc2_base.out 2>/dev/null; do sleep 20; done
run() { name=$1; shift; $PY runs/tactical/make_hc2.py $name "$@" >/dev/null && bash runs/tactical/run_hc2.sh $name; }
run skiponly   DIRECTION_HEAD_MODE=skip_only
run ep3        EPOCHS=3
run lr3e4      LR=0.0003
run dir5       LAMBDA_DIR=5.0
run look20     LOOKBACK=20
run nophys     LAMBDA_T_PERP=0 LAMBDA_CASIMIR=0 LAMBDA_HD=0 LAMBDA_IFE=0 LAMBDA_VAC=0 LAMBDA_VAC_OVERFLOW=0
