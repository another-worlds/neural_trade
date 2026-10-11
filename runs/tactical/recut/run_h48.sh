export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut/runs/tactical/geom2; L=../common/lowrun.py
$PY $L holdout_min.py > holdout_min_recut.out 2>&1
$PY $L holdout_min_nextopen.py > holdout_min_nextopen_recut.out 2>&1
