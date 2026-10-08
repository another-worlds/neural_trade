#!/bin/bash
# Order (lead, 2026-10-08): ind1 -> hc6 (hypotheses 10-13 on the no-price base) -> ind2 (indicator starting sets).
cd /d/nt/nt_tactical
until grep -q "^done ind1" runs/tactical/ind/run_ind1.out 2>/dev/null; do sleep 30; done
echo "--- hc6 (hypotheses 10-13) start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/hc6/run_guarded_hc6.sh runs/tactical/hc6/tasks.txt "done hc6" > runs/tactical/hc6/run.out 2>&1
echo "--- ind2 (indicator starting parameters, tiny runs) start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/ind/run_guarded_ind2.sh runs/tactical/ind/tasks_ind2.txt "done ind2" > runs/tactical/ind/run_ind2.out 2>&1
