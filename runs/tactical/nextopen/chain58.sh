PY=/c/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_wt_nextopen/runs/tactical/nextopen
export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2
$PY run_c.py > run_c.out 2>&1
$PY run_a.py > run_a.out 2>&1
$PY run_b.py 1h > run_b_1h.out 2>&1
$PY run_b.py 1m > run_b_1m.out 2>&1
