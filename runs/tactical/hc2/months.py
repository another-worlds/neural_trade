"""Month sets of the hill-climb v2 (SPEC.md): the lockbox draw, the dev months, the warm-up month. Nothing here reads data."""
import numpy as np
import pandas as pd

SEED = 20261013
FIRST, LAST = "2019-01", "2024-06"        # 2024-07 onward is a spent period: out of the climb and the lockbox entirely
WARMUP = "2018-12"                         # predicted (never scored): its confidences give January 2019's threshold
PER_YEAR = 2


def draw_lockbox(seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for y in range(2019, 2025):
        avail = [f"{y}-{m:02d}" for m in range(1, 13) if FIRST <= f"{y}-{m:02d}" <= LAST]
        out += sorted(rng.choice(avail, PER_YEAR, replace=False).tolist())
    return out


LOCKBOX = tuple(draw_lockbox())
ALL_MONTHS = tuple(str(m) for m in pd.period_range(FIRST, LAST, freq="M"))
DEV = tuple(m for m in ALL_MONTHS if m not in LOCKBOX)


def month_bounds(m):
    p = pd.Period(m, "M")
    return p.start_time.value, (p + 1).start_time.value


def ref_month(m, excluded):
    """The calendar month before m, or the nearest earlier month that is not in `excluded` (a lockbox month during the climb)."""
    p = pd.Period(m, "M") - 1
    while str(p) in excluded:
        p = p - 1
    return str(p)
