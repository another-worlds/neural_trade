#!/bin/bash
# Resource-guarded launcher (owner, 2026-10-07: continue, and keep my processes from loading CPU and RAM critically).
# usage: run_guarded.sh <tasks-file> <done-marker>
#   tasks-file: one "spec-name shard" per line (e.g. "hc4_calval 0/3").
# A new task starts only when: fewer than MAX_PROCS of our tasks run, free RAM >= START_FREE_GB and CPU load < START_CPU.
# Emergency: if free RAM falls below KILL_FREE_GB, the newest of our screen processes is stopped (its task is put back
# at the end of the queue; screen resume is idempotent, finished trials are kept). Every decision is logged to guard.log.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8
PY=C:/Users/Step/miniforge3/envs/nt/python
MAX_PROCS=3; START_FREE_GB=12; START_CPU=85; KILL_FREE_GB=5
LOG=runs/tactical/hc4/guard.log
mapfile -t QUEUE < "$1"
free_gb() { wmic OS get FreePhysicalMemory 2>/dev/null | grep -o '[0-9]\+' | head -1 | awk '{printf "%d", $1/1048576}'; }
cpu() { wmic cpu get LoadPercentage 2>/dev/null | grep -o '[0-9]\+' | awk '{s+=$1;n++} END{printf "%d", n?s/n:0}'; }
ours() { wmic process where "name='python.exe' and CommandLine like '%configs/tactical/hc4_%'" get ProcessId,CreationDate 2>/dev/null | grep -o '[0-9]\{14\}[^ ]* *[0-9]\+' ; }
declare -A PIDTASK
launch() { set -- $1; n=$1; sh=$2; tag=${sh//\//of}
  ( $PY -m neural_trade.cli screen configs/tactical/$n.yaml --store runs/tactical --shard $sh >> runs/tactical/hc4/${n}_${tag}.log 2>&1; echo "$(date +%T) exit $? $n $sh" >> $LOG ) &
  PIDTASK[$!]="$n $sh"; echo "$(date +%T) start $n $sh (free $(free_gb) GB, cpu $(cpu)%)" >> $LOG; }
while [ ${#QUEUE[@]} -gt 0 ] || [ -n "$(jobs -rp)" ]; do
  f=$(free_gb); c=$(cpu); r=$(jobs -rp | wc -l)
  if [ "$f" -lt "$KILL_FREE_GB" ] && [ "$r" -gt 0 ]; then
    newest=$(ours | sort | tail -1 | awk '{print $2}')
    if [ -n "$newest" ]; then
      echo "$(date +%T) RAM low ($f GB): stop pid $newest" >> $LOG; taskkill //PID $newest //F >/dev/null 2>&1
      for j in "${!PIDTASK[@]}"; do if ! kill -0 $j 2>/dev/null; then QUEUE+=("${PIDTASK[$j]}"); unset PIDTASK[$j]; fi; done
    fi
  elif [ ${#QUEUE[@]} -gt 0 ] && [ "$r" -lt "$MAX_PROCS" ] && [ "$f" -ge "$START_FREE_GB" ] && [ "$c" -lt "$START_CPU" ]; then
    launch "${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}"); sleep 45   # let the new process load its data before the next check
    continue
  fi
  sleep 20
done
echo "$2"
