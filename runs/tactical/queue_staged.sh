#!/bin/bash
# Staged queue (replaces queue_r2/r3). Stage 1: chunk c0 of every variant; gate; stage 2: the rest for the promoted.
cd /d/nt/nt_tactical
PY=C:/Users/Step/miniforge3/envs/nt/python
while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -q "hc2_base_c2"; do sleep 20; done  # let base c2 finish
mk() { name=$1; shift; $PY runs/tactical/make_hc2.py $name "$@" >/dev/null; }
mk skiponly DIRECTION_HEAD_MODE=skip_only
mk ep3 EPOCHS=3
mk lr3e4 LR=0.0003
mk dir5 LAMBDA_DIR=5.0
mk look20 LOOKBACK=20
mk nophys LAMBDA_T_PERP=0 LAMBDA_CASIMIR=0 LAMBDA_HD=0 LAMBDA_IFE=0 LAMBDA_VAC=0 LAMBDA_VAC_OVERFLOW=0
mk calval CALIBRATE=true CALIB_MODE=value
mk calgrad CALIBRATE=true CALIB_MODE=gradient
VARS="skiponly ep3 lr3e4 dir5 look20 nophys calval calgrad"
for v in $VARS; do bash runs/tactical/run_chunks.sh $v c0; done
$PY runs/tactical/stage1_gate.py $VARS | tee runs/tactical/stage1_gate.out
bash runs/tactical/run_chunks.sh base c3 c4
for v in $(grep PROMOTE runs/tactical/stage1_gate.out | cut -d: -f1); do bash runs/tactical/run_chunks.sh $v c1 c2 c3 c4; done
echo done staged
