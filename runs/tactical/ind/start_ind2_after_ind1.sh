#!/bin/bash
cd /d/nt/nt_tactical
until grep -q "^done ind1" runs/tactical/ind/run_ind1.out 2>/dev/null; do sleep 30; done
echo "--- ind2 (indicator starting parameters, tiny runs) start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/ind/run_guarded_ind2.sh runs/tactical/ind/tasks_ind2.txt "done ind2"
