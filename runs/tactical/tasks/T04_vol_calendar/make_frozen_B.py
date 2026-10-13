"""T04 Part B freeze: fit once on the Binance file 2025-10-02..2026-09-30 and write frozen_B.json. Reads no bar after 2026-09-30 23:59."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t04_lib as L  # noqa: E402

if __name__ == "__main__":
    bars = L.load_freeze_bars(L.T.FWD_CSV); print("bars", len(bars), bars.index.min(), bars.index.max())
    art = L.freeze(bars); json.dump(art, open(f"{HERE}/frozen_B.json", "w"), indent=1)
    print({k: v for k, v in art.items() if k != "models"})
