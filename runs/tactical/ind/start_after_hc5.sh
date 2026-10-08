#!/bin/bash
# The tiny-run indicator experiment starts when the no-price experiment is done.
cd /d/nt/nt_tactical
until grep -q "^done hc5" runs/tactical/hc4/run_hc5.out 2>/dev/null; do sleep 30; done
echo "--- ind1 (indicators, tiny runs) start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/ind/run_guarded_ind1.sh runs/tactical/ind/tasks_ind1.txt "done ind1"
