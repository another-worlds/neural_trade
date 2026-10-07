#!/bin/bash
# Resource-guarded launcher, v2 (owner, 2026-10-07: "control in real time that you do not cause an overload").
# usage: run_guarded.sh <tasks-file> <done-marker>      tasks-file: one "spec-name shard" per line.
# Every CHECK seconds: free RAM, CPU load, GPU use and our process count go to resources.csv (shown on the dashboard).
#  - Our python processes run at BELOW-NORMAL priority, so the machine and other work stay responsive.
#  - Start a task only if: < MAX_PROCS of ours run, free RAM >= START_FREE_GB, CPU < START_CPU.
#  - Stop the newest of ours (and re-queue its task; resume keeps finished trials) if free RAM < KILL_FREE_GB,
#    or if CPU >= HOT_CPU for HOT_CHECKS checks in a row while 2+ of ours run.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
MAX_PROCS=3; START_FREE_GB=12; START_CPU=80; KILL_FREE_GB=6; HOT_CPU=95; HOT_CHECKS=6; CHECK=10
LOG=runs/tactical/hc4/guard.log; CSV=runs/tactical/hc4/resources.csv
[ -f $CSV ] || echo "time,free_gb,cpu,gpu_util,gpu_mem_mb,ours,action" > $CSV
mapfile -t QUEUE < "$1"
free_gb() { wmic OS get FreePhysicalMemory 2>/dev/null | grep -o '[0-9]\+' | head -1 | awk '{printf "%.1f", $1/1048576}'; }
cpu() { wmic cpu get LoadPercentage 2>/dev/null | grep -o '[0-9]\+' | awk '{s+=$1;n++} END{printf "%d", n?s/n:0}'; }
gpu() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' ; }
ours_pids() { wmic process where "name='python.exe' and CommandLine like '%configs/tactical/hc4_%'" get ProcessId 2>/dev/null | grep -o '[0-9]\+'; }
newest() { wmic process where "name='python.exe' and CommandLine like '%configs/tactical/hc4_%'" get CreationDate,ProcessId 2>/dev/null | grep -o '^[0-9]\{14\}[^ ]* *[0-9]\+' | sort | tail -1 | awk '{print $2}'; }
declare -A JOBTASK; hot=0
launch() { set -- $1; n=$1; sh=$2; tag=${sh//\//of}
  ( $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical --shard $sh >> runs/tactical/hc4/${n}_${tag}.log 2>&1; echo "$(date +%T) exit $? $n $sh" >> $LOG ) &
  JOBTASK[$!]="$n $sh"; echo "$(date +%T) start $n $sh" >> $LOG; }
requeue_dead() { for j in "${!JOBTASK[@]}"; do if ! kill -0 $j 2>/dev/null; then
  t="${JOBTASK[$j]}"; unset JOBTASK[$j]
  if grep -q "exit [1-9][0-9]* ${t}\$" $LOG || [ "$STOPPED" = "$j" ]; then QUEUE+=("$t"); echo "$(date +%T) requeue $t" >> $LOG; fi; fi; done; STOPPED=; }
while [ ${#QUEUE[@]} -gt 0 ] || [ -n "$(jobs -rp)" ]; do
  for p in $(ours_pids); do wmic process where "ProcessId=$p" CALL setpriority 16384 >/dev/null 2>&1; done   # below normal
  f=$(free_gb); c=$(cpu); g=$(gpu); n=$(ours_pids | wc -l); act=""
  if [ "$c" -ge "$HOT_CPU" ]; then hot=$((hot+1)); else hot=0; fi
  if { awk "BEGIN{exit !($f < $KILL_FREE_GB)}" || { [ "$hot" -ge "$HOT_CHECKS" ] && [ "$n" -ge 2 ]; }; } && [ "$n" -gt 0 ]; then
    p=$(newest); act="stop $p (free ${f} GB, cpu ${c}%)"; echo "$(date +%T) $act" >> $LOG
    for j in "${!JOBTASK[@]}"; do :; done
    taskkill //PID $p //F >/dev/null 2>&1; hot=0; sleep 5
    for j in "${!JOBTASK[@]}"; do kill -0 $j 2>/dev/null || STOPPED=$j; done; requeue_dead
  else
    requeue_dead
    if [ ${#QUEUE[@]} -gt 0 ] && [ "$n" -lt "$MAX_PROCS" ] && awk "BEGIN{exit !($f >= $START_FREE_GB)}" && [ "$c" -lt "$START_CPU" ]; then
      launch "${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}"); act="start"
    fi
  fi
  echo "$(date +%T),$f,$c,$g,$n,$act" >> $CSV
  [ "$act" = "start" ] && sleep 40 || sleep $CHECK   # a new process loads its data first
done
echo "$2"
