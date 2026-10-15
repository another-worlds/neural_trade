"""O11 part a freeze: fits ONCE on the Binance bars 2025-10-01 .. 2026-09-30 23:59 and writes frozen_P_a.json (+ sha256). Refuses if the file exists."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import o11_freeze as Z  # noqa: E402
out = os.path.join(HERE, "frozen_P_a.json")
if os.path.exists(out):
    raise SystemExit("REFUSED: frozen_P_a.json exists; the freeze is made once.")
from common_run import nice; nice(); t0 = time.time()
bars = Z.P5.load_freeze_bars(); assert bars.index.max() <= Z.P5.B_HI and bars.index.min() >= Z.P5.B_LO
art = Z.freeze_a(bars)
json.dump(art, open(out, "w"), indent=1); open(os.path.join(HERE, "frozen_P_a.sha256"), "w").write(art["sha256"] + "\n")
print("FROZEN", art["sha256"], "bars", art["n_bars"], art["first_bar"], art["last_bar"], f"{time.time() - t0:.0f}s")
