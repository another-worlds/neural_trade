"""Path set-up and the imports of the existing tactical libraries, used unchanged (T01 targets/S3, T11 V3 + fit pattern, T10 H3 + labels +
day-block bootstrap, timefix clocks). numpy / pandas / sklearn / scipy only; works from either env."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T14 = os.path.dirname(HERE)
TASKS = os.path.dirname(T14)
TAC = os.path.dirname(TASKS)
ROOT = os.path.dirname(os.path.dirname(TAC))
for _p in (os.path.join(TAC, "common"), os.path.join(TASKS, "T01_vol_forward"), os.path.join(TASKS, "T11_utc_seasonality"),
           os.path.join(TASKS, "T10_barrier_hit"), os.path.join(TASKS, "T03_vol_intervals"), os.path.join(TASKS, "T09_regression_anatomy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import timefix as TF  # noqa: E402,F401
import t01_lib as T01  # noqa: E402,F401
import t11_lib as T11  # noqa: E402,F401
import t10_lib as T10  # noqa: E402,F401
import t03_lib as T3  # noqa: E402,F401
