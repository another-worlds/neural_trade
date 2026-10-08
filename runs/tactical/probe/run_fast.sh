#!/bin/bash
# Fast probe (owner: "not tactical timescales"): one run, 3 epochs, probe every 20 steps, BELOW-NORMAL priority (not idle),
# 9 GB GPU cap, one process. Writes "done probe" so the queued no-price experiment starts after it.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8 TF_FORCE_GPU_ALLOW_GROWTH=true NT_GPU_MEMORY_LIMIT_MB=9000
PY=C:/Users/Step/miniforge3/envs/nt/python
$PY runs/tactical/probe/probe_run.py 2021-10-13T18:00:00 0 EPOCHS=3 PROBE_EVERY=20 > runs/tactical/probe/log_fast_2021-10-13_0.txt 2>&1 &
pid=$!; sleep 3; for p in $(wmic process where "name='python.exe' and CommandLine like '%probe_run.py%'" get ProcessId 2>/dev/null | grep -o '[0-9]\+'); do wmic process where "ProcessId=$p" CALL setpriority 16384 >/dev/null 2>&1; done
wait $pid; echo "exit $?"
echo done probe
