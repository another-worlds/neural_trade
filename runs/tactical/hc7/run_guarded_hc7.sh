#!/bin/bash
# Resource-guarded launcher, v4 = v3 with 2 processes, each capped at 4.5 GB GPU memory (owner: use the free capacity). v3 (owner, 2026-10-07: "everything lags" -> "launch, only more carefully").
# v2 let 3 processes fill the GPU (100% use, 11.9 of 12.3 GB) - the same card draws the desktop - and missed it.
# usage: run_guarded_v3.sh <tasks-file> <done-marker>      tasks-file: one "spec-name shard" per line.
# Every CHECK s: free RAM, CPU, GPU use and free GPU memory, our process count -> resources.csv (dashboard panel).
#  - ONE training process at a time (MAX_PROCS=1), at BELOW-NORMAL priority (16384; owner 2026-10-08): Windows always serves the owner's work first.
#  - TensorFlow takes GPU memory only as it needs it (TF_FORCE_GPU_ALLOW_GROWTH) instead of the whole card.
#  - Start only if: free RAM >= 16 GB, free GPU memory >= 5 GB, GPU use < 40%, CPU < 95%.
#  - Stop ours and re-queue (resume keeps finished trials) if: free RAM < 8 GB, or free GPU memory < 1.5 GB, or
#    CPU >= 98% or GPU use >= 98% for HOT_CHECKS checks in a row.
cd /d/nt/nt_tactical
export PYTHONPATH=D:/nt/nt_tactical/src PYTHONIOENCODING=utf-8 TF_FORCE_GPU_ALLOW_GROWTH=true NT_GPU_MEMORY_LIMIT_MB=4500
PY=C:/Users/Step/miniforge3/envs/nt/python
MAX_PROCS=2; START_RAM=16; START_VRAM_MB=5000; START_GPU=40; START_CPU=95; KILL_RAM=8; KILL_VRAM_MB=1500; HOT=98; HOT_CHECKS=6; CHECK=10
LOG=runs/tactical/hc4/guard.log; CSV=runs/tactical/hc4/resources.csv
[ -f $CSV ] || echo "time,free_gb,cpu,gpu_util,gpu_mem_mb,ours,action" > $CSV
mapfile -t QUEUE < "$1"
free_gb() { wmic OS get FreePhysicalMemory 2>/dev/null | grep -o '[0-9]\+' | head -1 | awk '{printf "%.1f", $1/1048576}'; }
cpu() { wmic cpu get LoadPercentage 2>/dev/null | grep -o '[0-9]\+' | awk '{s+=$1;n++} END{printf "%d", n?s/n:0}'; }
gq() { nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
ours_pids() { wmic process where "name='python.exe' and CommandLine like '%configs/tactical/hc7_%'" get ProcessId 2>/dev/null | grep -o '[0-9]\+'; }
declare -A JOBTASK; hotc=0; STOPPED=
launch() { local t="$1" sn sh tag; set -- $t; sn=$1; sh=$2; tag=${sh//\//of}
  ( $PY -m neural_trade.cli screen configs/tactical/$sn.yaml --store runs/tactical --shard $sh >> runs/tactical/hc4/${sn}_${tag}.log 2>&1; echo "$(date +%T) exit $? $sn $sh" >> $LOG ) &
  JOBTASK[$!]="$t"; echo "$(date +%T) start $t" >> $LOG; }
reap() { local j t; for j in "${!JOBTASK[@]}"; do if ! kill -0 $j 2>/dev/null; then t="${JOBTASK[$j]}"; unset JOBTASK[$j]
  if [ -n "$STOPPED" ] || grep -q "exit [1-9][0-9]* ${t}\$" $LOG; then QUEUE+=("$t"); echo "$(date +%T) requeue $t" >> $LOG; fi; fi; done; STOPPED=; }
while [ ${#QUEUE[@]} -gt 0 ] || [ -n "$(jobs -rp)" ]; do
  for p in $(ours_pids); do wmic process where "ProcessId=$p" CALL setpriority 16384 >/dev/null 2>&1; done
  f=$(free_gb); c=$(cpu); IFS=, read gu gm gt <<< "$(gq)"; vfree=$((gt - gm)); k=$(ours_pids | wc -l); act=""
  if [ "$c" -ge "$HOT" ] || [ "$gu" -ge "$HOT" ]; then hotc=$((hotc+1)); else hotc=0; fi
  if [ "$k" -gt 0 ] && { awk "BEGIN{exit !($f < $KILL_RAM)}" || [ "$vfree" -lt "$KILL_VRAM_MB" ] || [ "$hotc" -ge "$HOT_CHECKS" ]; }; then
    for p in $(ours_pids); do taskkill //PID $p //F >/dev/null 2>&1; done
    act="stop (free ${f} GB, VRAM free ${vfree} MB, cpu ${c}%, gpu ${gu}%)"; echo "$(date +%T) $act" >> $LOG
    STOPPED=1; hotc=0; sleep 5; reap; sleep 60   # cool down before trying again
  else
    reap
    if [ ${#QUEUE[@]} -gt 0 ] && [ "$k" -lt "$MAX_PROCS" ] && awk "BEGIN{exit !($f >= $START_RAM)}" && [ "$vfree" -ge "$START_VRAM_MB" ] && [ "$gu" -lt "$START_GPU" ] && [ "$c" -lt "$START_CPU" ]; then
      launch "${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}"); act="start"
    fi
  fi
  echo "$(date +%T),$f,$c,$gu,$gm,$k,$act" >> $CSV
  if [ "$act" = "start" ]; then sleep 40; else sleep $CHECK; fi
done
echo "$2"
