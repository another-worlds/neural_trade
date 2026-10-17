#!/bin/bash
cd "$(dirname "$0")/../core"
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
PY=D:/nt/nt_mm_env/python.exe
R=../M4_regime_card
for v in M4_img M4_cap M4_both; do
  $PY evaluate.py --view $v --c2-view M4_num --out $R/$v > $R/eval_$v.log 2>&1
done
$PY $R/build_m4.py --stage gbm > $R/build_gbm.log 2>&1
$PY evaluate.py --view M4b_img_gbm --c2-view M4_num --out $R/M4b_img_gbm > $R/eval_M4b_img_gbm.log 2>&1
echo done > $R/run_all.done
