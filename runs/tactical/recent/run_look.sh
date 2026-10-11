export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=/c/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_tactical_recent/runs/tactical/recent
$PY wf.py run 1h look > look_1h.log 2>&1
$PY wf.py run 1m look > look_1m.log 2>&1
