"""Walk-forward folds (SPEC 'walk_forward'): annual expanding windows, model frozen inside each test block."""
import datetime as dt

DAY = 86400


def ts(y, m=1, d=1):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


TRAIN_START = {1: ts(2020, 2, 1), 2: ts(2020, 2, 26)}           # track -> first training row (warm-ups)
SEAL_START = ts(2025, 11, 1)
SEAL_END = ts(2026, 10, 31)
FOLDS = [
    ("Y2021", ts(2021), ts(2022)),
    ("Y2022", ts(2022), ts(2023)),
    ("Y2023", ts(2023), ts(2024)),
    ("Y2024", ts(2024), ts(2025)),
    ("Y2025", ts(2025), SEAL_START),
]
FOLD_S = ("S", SEAL_START, SEAL_END)
DEV_OOS = (ts(2021), SEAL_START)
HALVES = [(ts(2021), ts(2023, 6, 2)), (ts(2023, 6, 2), SEAL_START)]


def train_range(track, test_start, h_minutes):
    """Training rows [start, end): labels of the last row (t + h + 1 bar) end before the test block."""
    return TRAIN_START[track], test_start - (h_minutes + 1) * 60


def inner_split(track, test_start, h_minutes):
    """Networks: inner validation = last 20% of the training window (>= 120 d, <= 365 d), 1-day purge both sides."""
    a, b = train_range(track, test_start, h_minutes)
    span = b - a
    v = min(max(int(0.2 * span), 120 * DAY), 365 * DAY)
    v_start = b - v
    return (a, v_start - DAY), (v_start, b)
