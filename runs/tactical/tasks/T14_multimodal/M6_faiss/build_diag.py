"""POST HOC diagnostics (not part of the pre-registered result): which part of the M6 view carries the increment?
M6x_keys: the 12 descriptor columns themselves (no retrieval) on the rows M6_faiss serves; M6x_dist: knn_dist_mean alone; M6x_labels: the three label statistics without the distance."""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m6_lib as M  # noqa
import anchors as AN, view_cache as VC  # noqa
for src in ("cut", "fwd"):
    v = VC.read_view("M6_faiss", src); d = pd.read_parquet(os.path.join(AN.CACHE_DIR, f"desc_{src}.parquet")).reindex(v.index)
    VC.write_view("M6x_keys", src, d[M.KEYS], note="post hoc diagnostic: the 12 descriptor keys, rows of M6_faiss")
    VC.write_view("M6x_dist", src, v[["knn_dist_mean"]], note="post hoc diagnostic: knn_dist_mean alone")
    VC.write_view("M6x_labels", src, v[["knn_dA_mean", "knn_dA_sd", "knn_hit50_mean"]], note="post hoc diagnostic: label stats without distance")
