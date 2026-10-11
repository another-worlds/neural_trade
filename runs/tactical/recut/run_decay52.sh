export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut/runs/tactical/lab; L=../common/lowrun.py
for q in build q1 q2 q3; do $PY $L decay_run.py $q > ../recut/decay/run_$q.log 2>&1; done
$PY $L decay_report.py > ../recut/decay/report.log 2>&1
