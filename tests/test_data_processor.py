"""Purged four-way split, scaler hygiene, window-relative input and target alignment (M3/M4)."""
from __future__ import annotations

import numpy as np

LOOKBACK, HORIZONS = 60, [10, 15, 20]


def test_purged_splits_leave_a_gap_between_every_pair_of_blocks(tf):
    from neural_trade.data.splits import make_purged_splits

    folds = make_purged_splits(6_000, lookback=LOOKBACK, horizon_steps=HORIZONS, n_folds=5)
    assert folds, "no fold had a non-empty training block"
    for f in folds:
        assert f.gap == LOOKBACK + max(HORIZONS) == 80
        blocks = [f.train, f.val, f.cal, f.test]
        assert all(len(b) > 0 for b in blocks)
        for earlier, later in zip(blocks, blocks[1:]):
            assert later[0] - earlier[-1] >= f.gap + 1
            # no bar is both a label of the earlier block and an input of the later block
            assert earlier[-1] + max(HORIZONS) - 1 < later[0] - LOOKBACK
    assert folds[-1].test[-1] == 6_000 - 1


def test_leak_count_without_gap_is_exactly_79(tf):
    """Regression pin for the leak the gap removes: LOOKBACK + max(H) - 1 = 79 sequences."""
    from neural_trade.data.splits import make_purged_splits

    f = make_purged_splits(6_000, lookback=LOOKBACK, horizon_steps=HORIZONS, n_folds=5, gap=0)[-1]
    label_bars = {i + h - 1 for i in f.train for h in HORIZONS}
    leaked = [j for j in f.val if any(b in label_bars for b in range(j - LOOKBACK, j))]
    assert len(leaked) == LOOKBACK + max(HORIZONS) - 1 == 79


def test_targets_are_strictly_forward_and_windows_end_at_last_close(tf, tiny_config, synthetic_close):
    from neural_trade.data.processor import DataProcessor

    dp = DataProcessor(tiny_config)
    X, y, lc, _ext = dp.make_sequences_with_extended_trends(synthetic_close, LOOKBACK)
    start = max(LOOKBACK, max(tiny_config.EXTENDED_TREND_PERIODS))
    rng = np.random.default_rng(0)
    for j in rng.integers(0, X.shape[0], size=25):
        anchor = start + j
        np.testing.assert_allclose(X[j], synthetic_close[anchor - LOOKBACK:anchor], rtol=0, atol=0)
        assert lc[j] == synthetic_close[anchor - 1]
        for k, h in enumerate(HORIZONS):
            assert anchor + h - 1 > anchor - 1  # label bar is after every input bar
            np.testing.assert_allclose(y[j, k], synthetic_close[anchor + h - 1] - synthetic_close[anchor - 1], atol=1e-2)


def test_prepare_datasets_scaler_on_train_only_and_window_relative_input(tf, tiny_config, tmp_path, synthetic_bars):
    from neural_trade.data.processor import DataProcessor

    cfg = tiny_config
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    cfg.CSV_PATH = str(csv)
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")

    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    (X_tr, y_tr_s, lc_tr, _e1, X_te, _y_te_s, lc_te, _e2, y_tr, y_te, scaler) = dp.prepare_datasets(df, close)

    # scaler statistics come from the TRAINING targets only
    np.testing.assert_allclose(scaler.mean_[0], y_tr.reshape(-1).mean(), rtol=1e-5)
    np.testing.assert_allclose(scaler.scale_[0], y_tr.reshape(-1).std(), rtol=1e-5)

    # four blocks in chronological order with the purge gap between them
    f = dp.fold
    assert f.train[-1] < f.val[0] < f.val[-1] < f.cal[0] < f.cal[-1] < f.test[0]
    assert f.val[0] - f.train[-1] >= f.gap + 1 and f.test[0] - f.cal[-1] >= f.gap + 1
    assert dp.val_block["X"].shape[0] == len(f.val) and dp.cal_block["X"].shape[0] == len(f.cal)
    assert (tmp_path / "scaler.joblib").exists() and not (tmp_path / "scaler_input.joblib").exists()

    # window-relative input: last element is the anchor close (0), and the raw window is recoverable
    np.testing.assert_allclose(X_tr[:, -1], 0.0, atol=1e-6)
    X_all = dp.make_sequences_with_extended_trends(close, cfg.LOOKBACK)[0]
    X_all = X_all[-cfg.MAX_SEQUENCE_COUNT:] if X_all.shape[0] > cfg.MAX_SEQUENCE_COUNT else X_all
    raw_train = X_all[f.train]
    np.testing.assert_allclose(X_tr * dp.input_scale + lc_tr[:, None], raw_train, atol=2.0)
    # level invariance: the same window shifted by a constant normalises identically
    shifted = ((raw_train + 5_000.0) - (lc_tr[:, None] + 5_000.0)) / dp.input_scale
    np.testing.assert_allclose(shifted, X_tr, atol=1e-3)
