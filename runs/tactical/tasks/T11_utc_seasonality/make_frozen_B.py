"""T11 freeze (SPEC.md): fit V0, V1, the chosen variant and V3 ONCE on the Binance file 2025-10-02..2026-09-30 and write frozen_B.json with its sha256.
Refuses unless selection.json is committed unchanged and the report-only results exist. Reads no bar after 2026-09-30 23:59."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t11_lib as L  # noqa: E402

if __name__ == "__main__":
    sel_path = os.path.join(HERE, "selection.json"); commit = L.assert_committed(sel_path); print("selection.json committed in", commit)
    assert os.path.exists(os.path.join(HERE, "results.json")), "run run_report.py (the report-only looks) before the freeze"
    sel = json.load(open(sel_path)); bars = L.load_freeze_bars(L.T.FWD_CSV); print("bars", len(bars), bars.index.min(), bars.index.max())
    art = L.freeze(bars, sel["chosen"], L.sha_file(sel_path)); json.dump(art, open(f"{HERE}/frozen_B.json", "w"), indent=1)
    print({k: v for k, v in art.items() if k != "models"})
