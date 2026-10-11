export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut/runs/tactical/recent; L=../common/lowrun.py
$PY $L feats.py build 1h > build_1h_recut.log 2>&1
$PY $L feats.py build 1m > build_1m_recut.log 2>&1
