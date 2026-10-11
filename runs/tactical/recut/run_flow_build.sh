export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut/runs/tactical/flow
$PY ../common/lowrun.py flowlib.py build 1m > build_1m_recut.log 2>&1
$PY ../common/lowrun.py flowlib.py build 5m > build_5m_recut.log 2>&1
