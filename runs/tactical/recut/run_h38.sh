export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=C:/Users/Step/miniforge3/envs/nt/python
cd /d/nt/nt_recut
H=runs/tactical/hourly/hourly.py; L=runs/tactical/common/lowrun.py
$PY $L $H --T 12 --tp 1.5 --sl 1.5 --feat rich --primary hgb --target barrier --meta mag --final
$PY $L $H --T 12 --tp 1.5 --sl 1.5 --feat tb7 --primary logreg --target sign --meta base --final
$PY $L $H --T 12 --tp 2.0 --sl 2.0 --feat tb7 --primary logreg --target barrier --meta base --final
