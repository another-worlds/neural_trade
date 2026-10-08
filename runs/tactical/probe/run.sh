#!/bin/bash
# Two probe runs, one at a time (owner's load rules: one process, idle priority, 9 GB GPU cap).
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8 TF_FORCE_GPU_ALLOW_GROWTH=true NT_GPU_MEMORY_LIMIT_MB=9000
PY=C:/Users/Step/miniforge3/envs/nt/python
for a in "2021-10-13T18:00:00 0" "2023-04-29T08:00:00 1"; do
  set -- $a
  $PY runs/tactical/probe/probe_run.py $1 $2 > runs/tactical/probe/log_$1_$2.txt 2>&1 &
  pid=$!; sleep 5; for p in $(wmic process where "name='python.exe' and CommandLine like '%probe_run.py%'" get ProcessId 2>/dev/null | grep -o '[0-9]\+'); do wmic process where "ProcessId=$p" CALL setpriority 64 >/dev/null 2>&1; done
  wait $pid; echo "exit $? $1 $2"
done
echo done probe
