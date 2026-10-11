"""Hard data cutoff for the recut (H66): no bar after CUTOFF is read anywhere.

CSV is a physical copy of Bitcoin_BTCUSDT.csv whose rows stop at CUTOFF (made once by ensure_cut_csv), so a script
that points at C.CSV cannot see a later bar. read_csv() additionally asserts the bound on whatever it returns.
"""
import os
import pandas as pd

CUTOFF = "2025-07-20 23:59"
SRC = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
CSV = "D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv"
LAST_MONTH = "2025-07"          # the look months are 2024-07 .. 2025-07 (the last one partial, to 07-20)
LOOK_END = "2025-07-20"


def ensure_cut_csv(src=SRC, dst=CSV, cutoff=CUTOFF):
    """Write dst = the header plus every row of src with timestamp <= cutoff (string compare is safe: ISO format)."""
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with open(src, "r") as f, open(tmp, "w", newline="") as g:
        g.write(f.readline())
        for line in f:
            if line[:16] > cutoff:          # 'YYYY-MM-DD HH:MM'
                break
            g.write(line)
    os.replace(tmp, dst)
    return dst


def assert_cut(x, name="data"):
    """x: DataFrame/Series/Index/array of timestamps (or a frame with a 'timestamp' column / DatetimeIndex)."""
    if isinstance(x, pd.DataFrame):
        x = x["timestamp"] if "timestamp" in x.columns else x.index
    t = pd.to_datetime(pd.Series(pd.Index(x).values)) if not isinstance(x, pd.Series) else pd.to_datetime(x)
    if len(t) and t.max() > pd.Timestamp(CUTOFF):
        raise AssertionError(f"{name}: bar {t.max()} is after the cutoff {CUTOFF}")
    return x


def read_csv(path=None, **kw):
    """pd.read_csv on the cut file with the bound checked (chunksize is passed through and checked per chunk)."""
    ensure_cut_csv()
    p = CSV if path in (None, SRC, CSV) else path
    if p != CSV:
        raise ValueError("cutoff.read_csv reads only the cut file")
    out = pd.read_csv(p, **kw)
    if kw.get("chunksize"):
        def gen():
            for ch in out:
                if "timestamp" in ch.columns:
                    assert_cut(ch["timestamp"], "chunk")
                yield ch
        return gen()
    if "timestamp" in out.columns:
        assert_cut(out["timestamp"], p)
    return out


ensure_cut_csv()
