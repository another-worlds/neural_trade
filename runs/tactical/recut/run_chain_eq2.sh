export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2
PY=C:/Users/Step/miniforge3/envs/nt/python; L=../common/lowrun.py
cd /d/nt/nt_recut/runs/tactical/recent; $PY $L analyze.py look > analyze_look_recut.out 2>&1
cd ../equity
$PY $L run_c.py > run_c.out 2>&1
$PY $L run_a.py > run_a.out 2>&1
$PY $L run_b.py 1h > run_b_1h.out 2>&1
$PY $L run_b.py 1m > run_b_1m.out 2>&1
$PY $L make_tables.py > make_tables.out 2>&1
