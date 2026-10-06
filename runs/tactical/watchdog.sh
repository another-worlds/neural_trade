#!/bin/bash
# Kills a tactical `cli screen` shard whose log has not changed for 6 minutes (a hung trial; the wrapper
# in run_hc2.sh then resumes it). Only touches processes whose command line contains configs/tactical/hc2_.
cd /d/nt/nt_tactical
while true; do
  now=$(date +%s)
  wmic process where "name='python.exe'" get ProcessId,CommandLine 2>/dev/null | grep "configs/tactical/hc2_" | while read -r line; do
    pid=$(echo "$line" | grep -o '[0-9]*[[:space:]]*$' | tr -d ' \r'); spec=$(echo "$line" | grep -o 'hc2_[a-z0-9_]*' | head -1); sh=$(echo "$line" | grep -o '[0-9]/3' | head -1 | cut -d/ -f1)
    log=runs/tactical/${spec}_${sh}.log
    [ -f "$log" ] || continue
    age=$(( now - $(date -r "$log" +%s) ))
    if [ "$age" -gt 360 ]; then echo "$(date +%T) kill $pid $spec shard $sh (log idle ${age}s)"; taskkill //PID $pid //F >/dev/null 2>&1; fi
  done
  sleep 60
done
