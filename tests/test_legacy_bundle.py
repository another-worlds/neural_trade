"""NT-047 repair: a serving bundle written before NT-047 still loads and predicts exactly as before.

tests/data/legacy_bundle_426de4f/ is the artifacts/ directory of the real run
runs/20260929T081632Z-426de4f-dirty-aba344d6 (trained at 426de4f, before NT-046 and NT-047: its
config.yaml has neither INPUT_SERIES nor INDICATOR_FAMILIES). reference_last700_at_54bdea8.npz holds
that bundle's predict_frame / predict_last output on the bundled CSV's last 700 rows, recorded with
the code of the NT-047 base commit 54bdea8 (D:/nt_scratch_047/legacy_ref.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).parent / "data" / "legacy_bundle_426de4f"
CSV = Path(__file__).resolve().parent.parent / "binance_btcusdt_1min_ccxt.csv"


def test_legacy_config_pins_the_pre_nt047_model(tmp_path):
    from neural_trade.training.artifacts import _pin_legacy_input
    from neural_trade.core.config import Config

    cfg = _pin_legacy_input(Config.from_yaml(BUNDLE / "config.yaml"), BUNDLE / "config.yaml")
    assert cfg.INPUT_SERIES == ["close"] and cfg.INDICATOR_FAMILIES == {}
    # an NT-046-era bundle records INDICATOR_FAMILIES: that value is kept
    text = (BUNDLE / "config.yaml").read_text(encoding="utf-8") + "INDICATOR_FAMILIES: {atr: [7]}\n"
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
    cfg46 = _pin_legacy_input(Config.from_yaml(tmp_path / "config.yaml"), tmp_path / "config.yaml")
    assert cfg46.INPUT_SERIES == ["close"] and cfg46.INDICATOR_FAMILIES == {"atr": [7]}
    # a key named only in a comment does not count as recorded (the YAML is parsed)
    text = (BUNDLE / "config.yaml").read_text(encoding="utf-8") + "# INPUT_SERIES was added by NT-047\n"
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
    assert _pin_legacy_input(Config.from_yaml(tmp_path / "config.yaml"),
                             tmp_path / "config.yaml").INPUT_SERIES == ["close"]
    # a current bundle is left as recorded
    Config().to_yaml(tmp_path / "config.yaml")
    assert _pin_legacy_input(Config(), tmp_path / "config.yaml").INPUT_SERIES == Config().INPUT_SERIES


@pytest.mark.data
def test_a_pre_nt047_bundle_loads_and_predicts_as_at_the_base_commit(tf):
    from neural_trade.serving.predictor import Predictor

    if not CSV.exists():
        pytest.skip("bundled CSV absent")
    ref = np.load(BUNDLE / "reference_last700_at_54bdea8.npz")
    p = Predictor.from_artifacts(BUNDLE)
    df = pd.read_csv(CSV).iloc[-700:]
    fr = p.predict_frame(df)
    # NT-204: a bundle with a calibration pipeline adds one text column per horizon (the signal state); the
    # numeric frame is the one recorded at the base commit
    sig_cols = [c for c in fr.columns if c.endswith("_direction_signal")]
    assert sig_cols == ["h0_direction_signal", "h1_direction_signal", "h2_direction_signal"]
    fr = fr.drop(columns=sig_cols)
    assert fr.shape == (641, 25)
    assert list(fr.columns) == [str(c) for c in ref["columns"]]
    # Same code path, same weights. The reference was recorded on the lead's machine, where the
    # result is bit-for-bit equal; CI's different BLAS (float32 GEMM tiling) moves it by round-off,
    # so the tolerance is sized for the frame's scales: prices ~1e5 (rtol 1e-4) and deltas of a
    # few dollars (atol 1e-3). A different model (the NT-047 OHLCV default loaded by mistake)
    # changes deltas by whole dollars and fails this.
    np.testing.assert_allclose(fr.to_numpy(np.float64), ref["values"], rtol=1e-4, atol=1e-3)
    last = p.predict_last(df["close"].to_numpy())
    np.testing.assert_allclose(last["h1"]["delta"], ref["last_h1_delta"][0], rtol=1e-4, atol=1e-3)
