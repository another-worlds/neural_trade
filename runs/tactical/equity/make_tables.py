"""Tables from the *.json results -> summary.txt. usage: python make_tables.py"""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); out = []
for f in sorted(glob.glob(f"{HERE}/[abc]_*.json")):
    d = json.load(open(f)); out.append(f"== {d['strategy']}\n   signals {d['n_signals']}, trades {d['n_trades']}, skipped while a position was open {d['n_skipped']}")
    out.append("cost  trades/day  hit     mean bps  cum ret %  maxDD %  longest DD d  months hit>60%   null cum% med/p95   null maxDD% med/p95")
    for c, v in d["levels"].items():
        s, n = v["stats"], v["null"]
        out.append(f"{c:>4s}  {s['trades_per_day']:9.2f}  {s['hit']:.4f}  {s['mean_bps']:+8.2f}  {s['cum_return_pct']:+9.1f}  {s['max_drawdown_pct']:7.1f}  {s['longest_drawdown_days']:12.0f}  "
                   f"{s['months_hit_above_60']:2d}/{s['n_months']:<2d} ({s['share_months_hit_above_60']:.0%}) [{s['monthly_hit_min']:.2f}-{s['monthly_hit_max']:.2f}]  "
                   f"{n['cum_return_pct_median']:+7.1f}/{n['cum_return_pct_p95']:+7.1f}   {n['max_drawdown_pct_median']:6.1f}/{n['max_drawdown_pct_p95']:6.1f}")
    out.append("")
open(f"{HERE}/summary.txt", "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out))
