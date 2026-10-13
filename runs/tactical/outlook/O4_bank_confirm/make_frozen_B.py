"""O4 Part B freeze: fit R0 and BANK once on the Binance file (first warm-up anchor 2025-10-05 .. 2026-09-30); reads no bar after 2026-09-30 23:59."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o4_lib as O  # noqa: E402
try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

if __name__ == "__main__":
    bars = O.load_freeze_bars(); print("bars", len(bars), bars.index.min(), bars.index.max())
    art = O.freeze(bars); json.dump(art, open(f"{HERE}/frozen_B.json", "w"), indent=1)
    print({k: v for k, v in art.items() if k != "states"})
    print({h: {k: v for k, v in s.items() if k in ("n_tr", "first_ta", "last_label_end")} for h, s in art["states"].items()})
    open(f"{HERE}/frozen_B.sha256", "w").write(art["sha256"] + "\n")
