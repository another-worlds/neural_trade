"""M6 tests (seconds, synthetic data). Run: D:/nt/nt_mm_env/python.exe -m pytest -q test_m6.py   (CUDA_VISIBLE_DEVICES=-1)"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m6_lib as M  # noqa: E402
import sr_lib as S  # noqa: E402


def synth_pool(n_days=60, seed=0):
    rng = np.random.default_rng(seed); ta = pd.date_range("2025-01-01", periods=n_days * 24, freq="h")
    desc = rng.normal(size=(len(ta), 5)); ok = rng.random(len(ta)) > 0.1; lab = rng.normal(size=(len(ta), 2))
    return ta, desc, ok, lab


def test_neighbours_never_from_after_ta_minus_one_day_and_labels_known():
    ta, desc, ok, lab = synth_pool()
    nbr, dist, pool_n = M.knn_pool_index(ta, ok, desc, kmax=50, min_pool=100)
    i, c = np.where(nbr >= 0); j = nbr[i, c]
    assert len(i) > 0
    assert (ta[j] + pd.Timedelta(minutes=60) <= ta[i] - pd.Timedelta(days=1)).all()            # the label window of j ended a day before ta
    assert (ta[j] < ta[i]).all() and (i != j).all()
    assert ok[j].all()                                                                          # index is a subset of the (labelled) ok anchors
    assert set(np.unique(j)) <= set(np.where(ok)[0])


def test_pool_size_rule_drops_rows():
    ta, desc, ok, lab = synth_pool(n_days=30)
    nbr, dist, pool_n = M.knn_pool_index(ta, ok, desc, kmax=20, min_pool=500)
    served = (nbr >= 0).all(1)
    assert not served[pool_n < 500].any()
    assert (pool_n[served] >= 500).all() and served.any() and (~served).any()


def test_perturbing_future_labels_or_descriptors_changes_nothing_in_the_past():
    ta, desc, ok, lab = synth_pool(); nbr, _, _ = M.knn_pool_index(ta, ok, desc, kmax=30, min_pool=100)
    cut = len(ta) // 2; d2 = desc.copy(); d2[cut:] += 5.0; ok2 = ok.copy(); ok2[cut:] = ~ok2[cut:]
    nbr2, _, _ = M.knn_pool_index(ta, ok2, d2, kmax=30, min_pool=100)
    # a query at index <= cut + 24 + 60 can only use pool rows < cut (label window ended a day before): neighbours unchanged for queries whose pool is all before cut
    safe = np.where(ta < ta[cut])[0]
    # day-batched standardisation uses the same pool, so identical for queries with day start - 1 day < ta[cut]
    assert (nbr[safe] == nbr2[safe]).all()


def test_neighbour_stats_match_manual_mean():
    ta, desc, ok, lab = synth_pool(); nbr, _, _ = M.knn_pool_index(ta, ok, desc, kmax=50, min_pool=100)
    mean, sd, served = M.neighbour_stats(nbr, 20, lab)
    i = np.where(served)[0][5]
    assert np.allclose(mean[i], lab[nbr[i, :20]].mean(0)) and np.allclose(sd[i], lab[nbr[i, :20]].std(0))
    assert np.isnan(mean[~served]).all()


def test_shuffle_control_really_permutes_labels_jointly_and_keeps_neighbours():
    ta, desc, ok, lab = synth_pool(); nbr, _, _ = M.knn_pool_index(ta, ok, desc, kmax=50, min_pool=100)
    mean, _, served = M.neighbour_stats(nbr, 50, lab)
    smean, _, served2 = M.neighbour_stats(nbr, 50, lab, ta=ta, shuffle=True, seed=3)
    assert (served == served2).all()
    assert not np.allclose(mean[served], smean[served])                                          # genuinely different values
    # per-day the shuffle is a permutation of the neighbour pool's labels: multiset of (label0, label1) pairs preserved
    days = ta.normalize().values; d = np.unique(days[served])[10]; qi = np.where(served & (days == d))[0]
    pool = np.unique(nbr[qi][:, :50]); rng = np.random.default_rng(3 + int(d.astype("datetime64[D]").astype(np.int64)))
    perm = pool[rng.permutation(len(pool))]
    assert sorted(perm.tolist()) == sorted(pool.tolist()) and (perm != pool).any()
    assert np.allclose(smean[qi], lab[perm[np.searchsorted(pool, nbr[qi][:, :50])]].mean(1))    # joint permutation of both label columns
    # the shuffled feature is uncorrelated with the real one beyond noise, but marginals agree
    assert abs(np.corrcoef(mean[served][:, 0], smean[served][:, 0])[0, 1]) < 0.3
    assert abs(mean[served].mean() - smean[served].mean()) < 0.05


def synth_bars(n=20 * 24 * 60, seed=1):
    rng = np.random.default_rng(seed); idx = pd.date_range("2025-03-01", periods=n, freq="min")
    c = 50000 * np.exp(np.cumsum(rng.normal(0, 4e-4, n))); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 1e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 1e-4, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.random(n)}, index=idx)


def test_descriptor_is_causal_bars_at_or_after_ta_change_nothing():
    m1 = synth_bars(); ta = pd.date_range("2025-03-17", periods=24, freq="h"); sig = np.full(len(ta), 0.004)
    X = M.descriptors(m1, ta, sig, S.sr_block, S.clean_windows)
    assert np.isfinite(X).all(1).sum() > 0
    t0 = ta[10]; m2 = m1.copy(); m2.loc[m2.index >= t0, ["open", "high", "low", "close"]] *= 1.37
    m2.loc[m2.index >= t0, "volume"] = 99.0
    X2 = M.descriptors(m2, ta, sig, S.sr_block, S.clean_windows)
    assert np.array_equal(np.nan_to_num(X[:11], nan=-9), np.nan_to_num(X2[:11], nan=-9))            # anchors up to t0 (inclusive) read only bars < t0
    assert not np.array_equal(np.nan_to_num(X[11:], nan=-9), np.nan_to_num(X2[11:], nan=-9))


def test_trailing_pct_is_causal_and_uniform():
    rng = np.random.default_rng(0); x = rng.normal(size=3000); p = M.trailing_pct(x, win=1000, min_n=100)
    x2 = x.copy(); x2[2000:] = 99.0; p2 = M.trailing_pct(x2, win=1000, min_n=100)
    assert np.array_equal(p[:2000], p2[:2000], equal_nan=True)
    assert np.isnan(p[:99]).all() and 0.3 < np.nanmean(p) < 0.7 and np.nanmin(p) >= 0 and np.nanmax(p) <= 1
