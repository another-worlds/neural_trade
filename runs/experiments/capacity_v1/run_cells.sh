#!/bin/bash
# NT-104 capacity_v1: runs the scenario one cell at a time, with the RUNBOOK GPU-free check before every
# launch (busy: wait, never run concurrently). Resumable: `scenario run` skips finished cells.
# Stops after 15 cells or when a cell run adds no new result (failure).
cd D:/nt/nt_wt_104ab || exit 1
export PYTHONPATH=D:/nt/nt_wt_104ab/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
LOG=D:/nt/nt_wt_104ab/runs/experiments/capacity_v1/logs
mkdir -p $LOG
count() { ls D:/nt/nt_wt_104ab/runs/scenarios/capacity_v1/*/result.json 2>/dev/null | wc -l; }
free() {
  out=$(nvidia-smi dmon -s um -c 10 | grep -v '^#')
  med=$(echo "$out" | awk '{print $2}' | sort -n | sed -n 5p)
  fb=$(echo "$out" | awk '{print $8}' | sort -n | tail -1)
  [ "$med" -le 30 ] && [ "$fb" -le 2000 ]
}
while [ "$(count)" -lt 15 ]; do
  until free; do echo "$(date -u +%FT%TZ) GPU busy, waiting" >> $LOG/driver.log; sleep 60; done
  before=$(count)
  echo "$(date -u +%FT%TZ) launching cell $((before+1))" >> $LOG/driver.log
  $PY -m neural_trade.cli scenario run configs/scenarios/capacity_v1.yaml --store D:/nt/nt_wt_104ab/runs --max-cells 1 >> $LOG/run.log 2>&1
  after=$(count)
  if [ "$after" -le "$before" ]; then echo "$(date -u +%FT%TZ) no new result, stopping" >> $LOG/driver.log; exit 2; fi
done
echo "$(date -u +%FT%TZ) all cells done" >> $LOG/driver.log
