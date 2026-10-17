"""T14 harness, part 2: the view-file contract.

Each candidate writes, per source, ONE file  D:/nt/nt_t14_cache/view_{name}_{source}.parquet  with
  * index  = the anchor timestamp (UTC, naive, name `ta`), a subset of anchors_{source}.parquet's index, unique;
  * columns = the view's numeric features (float, finite or NaN; at least one; unique non-empty string names).
A row missing from the file, or with any NaN in it, is dropped for BOTH the base and the candidate (the counts are reported). The reserved column
`point_log_rv` is the view's own point forecast of the next hour's log RV on T01's scale, ln(sum of the squared 1-minute log returns of the hour
+ 1e-12); when present it is also used for the 50/50 average with V3 (the prediction of A is V3's change forecast and `point_log_rv - LV0`, where
LV0 is the previous hour's log RV on the same scale). A sidecar view_{name}_{source}.meta.json records the columns, row counts, a free-text
`note` and the sha256 of the parquet.

    write_view(name, source, df, note="")     validates and writes
    read_view(name, source)                   the DataFrame as written
    align_view(df, anchors_index)             float matrix aligned to the anchors (NaN rows for anchors the file lacks)
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors as AN  # noqa: E402

POINT_COL = "point_log_rv"


class ViewContractError(ValueError):
    pass


def view_path(name, source):
    return os.path.join(AN.CACHE_DIR, f"view_{name}_{source}.parquet")


def validate_view(df, anchors_index):
    """Raise ViewContractError unless `df` honours the contract; returns df with a normalised index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ViewContractError("index must be a DatetimeIndex of anchor times (UTC, naive)")
    if df.index.tz is not None:
        raise ViewContractError("index must be tz-naive UTC")
    idx = df.index.astype("datetime64[ns]")
    if idx.has_duplicates:
        raise ViewContractError(f"{int(idx.duplicated().sum())} duplicated anchor timestamps")
    anc = pd.DatetimeIndex(anchors_index).astype("datetime64[ns]")
    extra = idx.difference(anc)
    if len(extra):
        raise ViewContractError(f"{len(extra)} rows are not anchors (first {extra[0]}); a view is aligned by anchor time")
    if df.shape[1] == 0:
        raise ViewContractError("no columns")
    if len(set(df.columns)) != df.shape[1] or any((not isinstance(c, str)) or c == "" for c in df.columns):
        raise ViewContractError("columns must be unique non-empty strings")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            raise ViewContractError(f"column {c!r} is not numeric")
    vals = df.to_numpy(float)
    if np.isinf(vals).any():
        raise ViewContractError("inf values (use NaN to drop a row)")
    out = df.astype(float).copy(); out.index = pd.DatetimeIndex(idx, name="ta")
    return out.sort_index()


def write_view(name, source, df, note=""):
    anc = AN.load_anchors(source).index
    out = validate_view(df, anc)
    os.makedirs(AN.CACHE_DIR, exist_ok=True); p = view_path(name, source); out.to_parquet(p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    ok = AN.load_anchors(source)["ok"]
    nan_rows = int(out.isna().any(axis=1).sum()); covered = int(out.reindex(anc[ok.values]).notna().all(axis=1).sum())
    meta = {"name": name, "source": source, "columns": list(out.columns), "n_rows": int(len(out)), "n_rows_with_nan": nan_rows,
            "n_ok_anchors": int(ok.sum()), "n_ok_anchors_covered": covered, "sha256": sha, "note": note}
    json.dump(meta, open(p.replace(".parquet", ".meta.json"), "w"), indent=1)
    return meta


def read_view(name, source):
    p = view_path(name, source)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_parquet(p); df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]"); df.index.name = "ta"
    return validate_view(df, AN.load_anchors(source).index)


def align_view(df, anchors_index):
    """(matrix [n_anchors, n_cols] float, column names): rows in anchor order, NaN where the view has no row."""
    anc = pd.DatetimeIndex(anchors_index).astype("datetime64[ns]")
    a = df.reindex(anc)
    return a.to_numpy(float), list(df.columns)


def view_sha(name, source):
    return hashlib.sha256(open(view_path(name, source), "rb").read()).hexdigest()
