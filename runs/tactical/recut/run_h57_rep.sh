export PYTHONIOENCODING=utf-8 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=3
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut/runs/tactical/volnet
$PY run.py --stage look --part 0 --nparts 1 --models ens --tag _recut_rep > log_look_recut_rep.txt 2>&1
