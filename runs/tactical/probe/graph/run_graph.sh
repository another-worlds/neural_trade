#!/bin/bash
cd /d/nt/nt_gprobe
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=D:/nt/nt_gprobe/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
git rev-parse HEAD
for de in 2021-10-13T18:00:00 2024-11-11T23:00:00; do
  $PY runs/tactical/probe/graph/gprobe_run.py $de 0 > runs/tactical/probe/graph/log_${de:0:10}_s0.txt 2>&1 &
  pid=$!; sleep 5
  for p in $(wmic process where "name='python.exe' and CommandLine like '%gprobe_run.py%'" get ProcessId 2>/dev/null | grep -o '[0-9]\+'); do wmic process where "ProcessId=$p" CALL setpriority 16384 >/dev/null 2>&1; done
  wait $pid; echo "exit $? $de"
done
echo done
