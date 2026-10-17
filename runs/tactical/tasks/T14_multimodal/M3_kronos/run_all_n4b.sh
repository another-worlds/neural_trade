#!/bin/bash
# post hoc M3b: 4 more independent paths per anchor (seed 777000), concatenated with the n4 paths -> N = 8
cd D:/nt/nt_t14_m3/runs/tactical/tasks/T14_multimodal/M3_kronos
export PYTHONIOENCODING=utf-8
for s in fwd cut; do D:/nt/nt_mm_legacy/python.exe run_kronos.py --source $s --n 4 --tag n4b --seed 777000 --amp bf16 --max-hours 1.2 2>&1 | grep -v ptxas; done
