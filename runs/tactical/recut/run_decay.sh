export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut/runs/tactical/lab; L=../common/lowrun.py
$PY $L decay.py > ../recut/lab/decay.out 2>&1
$PY $L decay_nextopen.py > ../recut/lab/decay_nextopen.out 2>&1
$PY $L decay_vol.py > ../recut/lab/decay_vol.out 2>&1
$PY $L decay_asym.py > ../recut/lab/decay_asym.out 2>&1
