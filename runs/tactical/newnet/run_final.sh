#!/bin/bash
# newnet2 step 3 (criterion 3): the held-out FINAL slices at 1 y, the regression and the chosen variant (3 seeds)
cd "$(dirname "$0")"
bash run_arm.sh final_linear --slices final --span 1y --arch linear --tag final_linear
bash run_arm.sh final_trainlin --slices final --span 1y --arch patch --train-lin --seeds 3 --tag final_trainlin
