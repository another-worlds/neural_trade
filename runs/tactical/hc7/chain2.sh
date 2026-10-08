#!/bin/bash
# Order (lead, 2026-10-08, after the owner's "check your three hypotheses"): hc6 (running) -> hc7 -> ind2.
cd /d/nt/nt_tactical
until grep -q "^done hc6" runs/tactical/hc6/run.out 2>/dev/null; do sleep 30; done
echo "--- hc7 (decisive indicators + hypotheses A, B) start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/hc7/run_guarded_hc7.sh runs/tactical/hc7/tasks.txt "done hc7" > runs/tactical/hc7/run.out 2>&1
echo "--- ind2 (indicator starting parameters, tiny runs) start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/ind/run_guarded_ind2.sh runs/tactical/ind/tasks_ind2.txt "done ind2" > runs/tactical/ind/run_ind2.out 2>&1
