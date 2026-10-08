#!/bin/bash
# The no-price experiment starts when the gradient probe has released the GPU (one owner-safe GPU job pattern).
cd /d/nt/nt_tactical
until grep -q "done probe" runs/tactical/probe/run.out 2>/dev/null; do sleep 30; done
echo "--- hc5 (no price: 3 vs 1 horizon) start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/hc4/run_guarded_v4_hc5.sh runs/tactical/hc4/tasks_hc5.txt "done hc5"
