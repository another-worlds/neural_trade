"""O5 freeze: fit B* and the candidate once on the Binance file (first warm-up anchor 2025-10-05 .. 2026-09-30); reads no bar after 2026-09-30 23:59.
The base is taken from choice.json (written by run_o5.py from the selection months only)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o5_lib as P  # noqa: E402
try:
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

if __name__ == "__main__":
    base = json.load(open(f"{HERE}/choice.json"))["base"]
    bars = P.load_freeze_bars(); print("bars", len(bars), bars.index.min(), bars.index.max())
    art = P.freeze(bars, base); json.dump(art, open(f"{HERE}/frozen_B.json", "w"), indent=1)
    print({k: v for k, v in art.items() if k not in ("states",)})
    print({h: {k: v for k, v in s.items() if k in ("n_tr", "first_ta", "last_label_end", "c")} for h, s in art["states"].items()})
    open(f"{HERE}/frozen_B.sha256", "w").write(art["sha256"] + "\n")
