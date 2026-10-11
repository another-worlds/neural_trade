export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut/runs/tactical/calendar
$PY ../common/lowrun.py run_calendar.py > run_recut.log 2>&1
