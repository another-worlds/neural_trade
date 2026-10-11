"""Seconds. The hard cutoff (no bar after 2025-07-20), causality of the study-3 features, the break-detection tools on known cases."""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaps_lib as G  # noqa: E402


def _csv(path, n=3000, start="2025-07-19 00:00"):
    ts = pd.date_range(start, periods=n, freq="min"); rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    pd.DataFrame({"timestamp": ts.astype(str), "open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)}).to_csv(path, index=False)


def test_loader_has_no_bar_after_the_cutoff(tmp_path):
    p = tmp_path / "b.csv"; _csv(p)
    s = G.load_bars(str(p))
    assert s.index.max() <= pd.Timestamp("2025-07-20 23:59") and s.index.max() == pd.Timestamp("2025-07-20 23:59")
    assert G.CUTOFF == pd.Timestamp("2025-07-20 23:59")


def test_loader_asserts_when_the_cutoff_is_passed_wrongly(tmp_path):
    p = tmp_path / "b.csv"; _csv(p)
    s = G.load_bars(str(p), cutoff="2025-07-19 12:00")
    assert s.index.max() <= pd.Timestamp("2025-07-19 12:00")


def test_real_file_stops_at_the_cutoff_if_present():
    if not os.path.exists(G.shapelib.CSV):
        return
    s = G.load_bars()
    assert s.index.max() <= G.CUTOFF


def test_study3_features_ignore_bars_from_the_anchor_on_and_targets_use_only_the_next_four_hours():
    import study3_vov as S3
    n = 9000; ts = pd.date_range("2025-01-01", periods=n, freq="min"); rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    df = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)}, index=ts)
    D, FS, TG, AR, ok = S3.build(df); k = 100; a = int(D.anch[k])
    df2 = df.copy(); cut = a   # change every bar from the anchor on
    df2.iloc[cut:, :4] = df2.iloc[cut:, :4].values * 1.7; df2.iloc[cut:, 4] = 5.0
    D2, FS2, TG2, AR2, ok2 = S3.build(df2)
    assert int(D2.anch[k]) == a
    for nm in FS:
        assert np.allclose(FS[nm][k], FS2[nm][k], equal_nan=True), nm
    assert not np.allclose(TG["A"][k], TG2["A"][k])
    # a bar beyond the target window (a+240 and later) does not change the targets
    df3 = df.copy(); df3.iloc[a + 240:, :4] = df3.iloc[a + 240:, :4].values * 1.7
    _, _, TG3, _, _ = S3.build(df3)
    assert np.isclose(TG["A"][k], TG3["A"][k]) and np.isclose(TG["B"][k], TG3["B"][k])


def test_break_search_finds_a_known_break_and_prefers_none_for_noise():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.full(40, 0.55), np.full(40, 0.50)]) + rng.normal(0, 0.005, 80)
    sel = G.select_k(y); assert sel["kstar"] == 1 and abs(sel["breaks"][0] - 40) <= 2 and sel["bic_k"][1] + 2 < sel["bic_trend"]
    _, _, p = G.chow_pvalue(y, nboot=200); assert p < 0.05
    z = rng.normal(0, 0.01, 80); sel0 = G.select_k(z); assert sel0["kstar"] == 0
    _, _, p0 = G.chow_pvalue(z, nboot=200); assert p0 > 0.05
    ci = G.break_intervals(y, 1, sel["breaks"], nboot=100); assert ci[0][0] <= sel["breaks"][0] <= ci[0][1]


def test_min_segment_length_is_respected():
    y = np.concatenate([np.zeros(3), np.ones(40), np.zeros(37)])
    for k, (_, br) in G.best_partitions(y, kmax=3, h=6).items():
        e = [0] + br + [len(y)]; assert all(b - a >= 6 for a, b in zip(e[:-1], e[1:]))
