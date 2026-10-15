"""T11 prospective scoring (SPEC.md), for a LATER session. usage: python score_B.py [--csv path] [--out-dir dir]

Loads frozen_B.json (refuses on a sha256 mismatch, or when selection.json no longer matches the sha it froze), reads the Binance file from
2026-09-28 (feature history), scores every frozen model on the complete months from 2026-10-02 with NO refit, and applies the criterion at the
looks of 3 and 6 complete months with Pocock alpha 0.0294 for each look. With fewer than 3 complete months it says so and writes nothing.
A month is complete when the file holds the 240 bars after its last minute (every anchor of the month exists). A look's file is never overwritten:
a later run writes resultsB_look<k>_followup.json instead. Nothing here is run in the freeze task (only on synthetic data in the tests)."""
import json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t11_lib as L  # noqa: E402
T = L.T

SCORE_FROM = pd.Timestamp("2026-09-28 00:00"); FIRST_MONTH = pd.Period("2026-10", freq="M"); COMPLETE_AFTER_MIN = 239


def load_frozen(path=os.path.join(HERE, "frozen_B.json"), selection_path=os.path.join(HERE, "selection.json")):
    art = json.load(open(path))
    if L.sha_of(art) != art["sha256"]:
        raise ValueError("frozen_B.json does not match its sha256")
    if os.path.exists(selection_path) and L.sha_file(selection_path) != art["selection_json_sha256"]:
        raise ValueError("selection.json differs from the file the freeze was made from")
    return art


def complete_months(last_bar, first=FIRST_MONTH):
    out = []; m = first
    while (m + 1).start_time + pd.Timedelta(minutes=COMPLETE_AFTER_MIN) <= last_bar:
        out.append(m); m = m + 1
    return out


def score(art, bars, months=None):
    months = complete_months(bars.index.max()) if months is None else months
    out = {"n_months": len(months), "months": [str(m) for m in months], "chosen": art["chosen"], "frozen_sha256": art["sha256"]}
    if len(months) < min(L.LOOKS):
        return {**out, "rows": [], "looks": {}, "status": f"withheld: {len(months)} complete months, need {min(L.LOOKS)}"}
    B = L.build(bars, 1441, "binance"); rows = [L.score_month(art["models"], B, m, m_start=L.SCORE_START) for m in months]
    looks = {str(k): L.verdict(rows, art["chosen"], k) for k in L.LOOKS if len(rows) >= k}
    comps = {v: L.paired(rows, v) for v in art["variants"] if v != "V0"} if len(rows) > 1 else {}
    return {**out, "rows": rows, "looks": looks, "comparisons": comps, "status": "verdict at look(s) " + ", ".join(looks)}


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    csv = arg("--csv", T.FWD_CSV); out_dir = arg("--out-dir", HERE)
    art = load_frozen(); print("frozen sha256", art["sha256"], "chosen", art["chosen"])
    bars = T.load_bars(csv, SCORE_FROM, pd.Timestamp("2100-01-01")); print("bars", len(bars), bars.index.min(), bars.index.max())
    res = score(art, bars); print(res["status"])
    for k, v in res["looks"].items():
        dst = os.path.join(out_dir, f"resultsB_look{k}.json")
        if os.path.exists(dst):
            dst = os.path.join(out_dir, f"resultsB_look{k}_followup.json")
        json.dump({**v, "months_rows": res["rows"][:int(k)], "frozen_sha256": art["sha256"]}, open(dst, "w"), indent=1)
        print(f"look {k}: primary", "PASS" if v["pass_primary"] else "FAIL", v["primary_A"], "| secondary", v["secondary_R"], "->", dst)
    for r in res["rows"]:
        print(r["month"], r["n_te"], " ".join(f"{k} {v:+.4f}" for k, v in r["skill_ar"].items()))


if __name__ == "__main__":
    main()
