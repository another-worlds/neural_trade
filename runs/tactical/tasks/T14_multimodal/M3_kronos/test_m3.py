"""M3 tests (CPU, mm_env, seconds): run  CUDA_VISIBLE_DEVICES=-1 D:/nt/nt_mm_env/python.exe -m pytest -q test_m3.py
1. no information after the anchor in the bars fed to Kronos (mutating every bar from the anchor on leaves the input identical, and it equals the saved input);
2. the saved inputs end before their anchor; 3. the view index is a subset of the anchors (and of the ok anchors); 4. feature maths on a hand-made path."""
import copy, os, sys
import numpy as np, pandas as pd, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core")); sys.path.insert(0, HERE)
import anchors as AN, view_cache as VC
import build_inputs as BI, make_view as MV

SRC = ["cut", "fwd"]


def _inputs(source):
    z = np.load(os.path.join(AN.CACHE_DIR, "m3", f"inputs_{source}.npz")); return z["ta_ns"].astype("datetime64[ns]"), z["x"], z["ts_ns"]


@pytest.mark.parametrize("source", SRC)
def test_mutating_bars_from_the_anchor_on_changes_nothing(source):
    st = AN.get_store(source); ta_all, X, TS = _inputs(source); rng = np.random.default_rng(0)
    for i in rng.choice(len(ta_all), 12, replace=False):
        ta = pd.Timestamp(ta_all[i]); ref = BI.anchor_input(source, ta)
        st2 = copy.copy(st); m1 = st.m1.copy(); m1.loc[m1.index >= ta, ["open", "high", "low", "close", "volume"]] = [1e9, 1e9, -1e9, 7e8, 1e12]
        st2.m1 = m1; st2._m5 = None
        mut = st2.bars_before(ta, BI.CTX, "5min")
        pd.testing.assert_frame_equal(ref, mut)
        assert np.array_equal(ref[["open", "high", "low", "close", "volume"]].to_numpy(), X[i]) and np.array_equal(ref.index.values.astype("datetime64[ns]").astype(np.int64), TS[i])
        # also bars from ta - 4 min (the last 5-min bin would be incomplete if it ended after ta) never leak: last bin closes at or before ta
        assert ref.index[-1] + pd.Timedelta(minutes=5) <= ta


@pytest.mark.parametrize("source", SRC)
def test_saved_inputs_end_before_the_anchor(source):
    ta, X, TS = _inputs(source)
    assert (TS[:, -1].astype("datetime64[ns]") < ta).all() and (TS[:, -1].astype("datetime64[ns]") + np.timedelta64(5, "m") <= ta).all()
    assert (np.diff(TS, axis=1) > 0).all()


@pytest.mark.parametrize("source", SRC)
def test_view_index_is_a_subset_of_the_ok_anchors(source):
    df = VC.read_view("M3_kronos", source); anc = AN.load_anchors(source)
    assert df.index.isin(anc.index).all() and anc.loc[df.index, "ok"].all() and not df.index.has_duplicates
    assert "point_log_rv" in df.columns and np.isfinite(df.to_numpy()).all()
    assert df["kr_phit50"].between(0, 1).all() and (df["kr_range_med"] > 0).all()


def test_path_features_by_hand():
    c0 = np.array([100.0]); P = np.zeros((1, 2, 12, 4))
    # path 0: flat at 100 except bar 3 high 100.6 (60 bps) ; path 1: closes drift 100 -> 100.12 (one 1% total is not reached)
    P[0, 0] = 100.0; P[0, 0, 3, 1] = 100.6
    P[0, 1] = 100.0; P[0, 1, :, 3] = 100.0 + np.arange(1, 13) * 0.01; P[0, 1, :, 1] = P[0, 1, :, 3]; P[0, 1, :, 0] = P[0, 1, :, 3]
    f = MV.path_features(P, c0)
    assert f["kr_phit50"][0] == 0.5                                                # only path 0 reaches 50 bps (high-based)
    assert np.isclose(f["kr_range_med"][0], np.median([0.6 / 100, 0.11 / 100 + 0.01 / 100 * 0 + 0.0]) if False else np.median([0.006, (100.12 - 100.0) / 100]), atol=1e-9)
    assert np.isclose(f["kr_ret_sd_across_paths"][0], np.std([0.0, np.log(100.12 / 100)], ddof=1))
    lr1 = np.diff(np.log(np.r_[100.0, 100.0 + np.arange(1, 13) * 0.01])); assert np.isclose(f["rv5_med"][0], np.median([0.0, (lr1 ** 2).sum()]))
