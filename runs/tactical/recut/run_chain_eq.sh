export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2
PY=C:/Users/Step/miniforge3/envs/nt/python; L=../common/lowrun.py
cd /d/nt/nt_recut/runs/tactical/equity
$PY $L run_c.py > run_c.out 2>&1
$PY $L run_a.py > run_a.out 2>&1
cd ../nextopen
$PY $L run_c.py > run_c.out 2>&1
$PY $L run_a.py > run_a.out 2>&1
