export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2
PY=C:/Users/Step/miniforge3/envs/nt/python; L=../common/lowrun.py
cd /d/nt/nt_recut/runs/tactical/recent
$PY $L wf.py run 1h look --workers 5 > look_1h.log 2>&1
$PY $L wf.py run 1m look --workers 5 > look_1m.log 2>&1
$PY $L analyze.py look > analyze_look_recut.out 2>&1
cd ../flow
$PY $L flowrun.py look --workers 4 > look_recut.log 2>&1
$PY $L analyze.py look > analyze_look_recut.out 2>&1
cd ../nextopen
$PY $L wf_no.py nextopen --workers 5 > h56_nextopen.log 2>&1
$PY $L wf_no.py close --workers 5 > h56_close.log 2>&1
$PY $L analyze_no.py > analyze_no.out 2>&1
