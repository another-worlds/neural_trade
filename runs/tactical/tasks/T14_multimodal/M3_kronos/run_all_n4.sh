#!/bin/bash
cd D:/nt/nt_t14_m3/runs/tactical/tasks/T14_multimodal/M3_kronos
export PYTHONIOENCODING=utf-8
for s in fwd cut; do D:/nt/nt_mm_legacy/python.exe run_kronos.py --source $s --n 4 --tag n4 --amp bf16 --max-hours 1.5 2>&1 | grep -v ptxas; done
