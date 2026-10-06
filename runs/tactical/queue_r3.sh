#!/bin/bash
# Round 3 queue (owner: "try loss balancing"): the existing calibration pass, value vs gradient (NT-101) mode.
# Starts after the round-2 queue is done. Fixed before any result.
cd /d/nt/nt_tactical
until grep -q "^done nophys" runs/tactical/queue_r2.out 2>/dev/null; do sleep 30; done
run() { name=$1; shift; C:/Users/Step/miniforge3/envs/nt/python runs/tactical/make_hc2.py $name "$@" >/dev/null && bash runs/tactical/run_hc2.sh $name; }
run calval  CALIBRATE=true CALIB_MODE=value
run calgrad CALIBRATE=true CALIB_MODE=gradient
echo done r3
