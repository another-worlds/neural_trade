"""T12 Look B freeze: fit H3 (T10 code) once per T on the Binance file 2025-10-02..2026-09-30 and write frozen_B.json (+ frozen_B.sha256).
Reads exactly the rows to 2026-09-30 23:59 (nrows), no October-2026 target."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t12_lib as T12

if __name__ == "__main__":
    bars = T12.load_freeze(); print("bars", len(bars), bars.index.min(), bars.index.max(), flush=True)
    art = T12.freeze(bars); json.dump(art, open(os.path.join(HERE, "frozen_B.json"), "w"), indent=1)
    open(os.path.join(HERE, "frozen_B.sha256"), "w").write(art["sha256"] + "\n")
    print({k: v for k, v in art.items() if k in ("m0", "first_month", "last_bar", "n_bars", "n_train", "train_label_end_max", "sha256")})
    for T in T12.TS:
        print(T, {X: (art["iso_first"][str(T)][str(X)]["n"], len(art["iso_first"][str(T)][str(X)]["x"])) for X in T12.XS})
