#!/bin/bash
# Start guard v4 only when none of our hc4 training processes runs (the v3 orphan finishes its shard first).
cd /d/nt/nt_tactical
while wmic process where "name='python.exe' and CommandLine like '%configs/tactical/hc4_%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; do sleep 20; done
echo "--- guard v4 start $(date +%T)" >> runs/tactical/hc4/guard.log
bash runs/tactical/hc4/run_guarded_v4.sh runs/tactical/hc4/tasks_r1c.txt "done hc4 r1"
