export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2
PY=C:/Users/Step/miniforge3/envs/nt/python; L=../common/lowrun.py
cd /d/nt/nt_recut/runs/tactical/nextopen
$PY $L run_c.py > run_c.out 2>&1
$PY $L run_a.py > run_a.out 2>&1
$PY $L run_b.py 1h > run_b_1h.out 2>&1
$PY $L run_b.py 1m > run_b_1m.out 2>&1
$PY $L analyze_no.py > analyze_no.out 2>&1
