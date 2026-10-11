"""H58 tables, close (equity/*.json, the recorded run) vs next-open (nextopen/*_nextopen.json) side by side, and the H38 held-out meta result.
usage: python make_tables_no.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); EQ = os.path.join(os.path.dirname(HERE), "equity"); out = []
pairs = [("a_h48_minute_barrier_meta", "(a) H48 minute triple barrier meta"), ("b_h56_1h_rich_l1_365", "(b) H56 1h rich:l1:365 top-10%"),
         ("b_h56_1m_all_l1_365", "(b) H56 1m all:l1:365 top-10%"), ("c_h38_hourly_barrier_meta", "(c) H38 hourly barrier meta")]
for tag, title in pairs:
    out.append(f"== {title}, 2024-07..2025-09")
    out.append("entry     cost trades  hit     mean bps  cum ret %  maxDD %  months hit>60%  null cum% med/p95   null maxDD% med/p95")
    for lab, path in (("close", f"{EQ}/{tag}.json"), ("nextopen", f"{HERE}/{tag}_nextopen.json")):
        d = json.load(open(path))
        for c, v in d["levels"].items():
            s, n = v["stats"], v["null"]
            out.append(f"{lab:9s} {c:>3s}  {s['n_trades']:5d}  {s['hit']:.4f}  {s['mean_bps']:+8.2f}  {s['cum_return_pct']:+9.1f}  {s['max_drawdown_pct']:7.1f}  "
                       f"{s['months_hit_above_60']:2d}/{s['n_months']:<2d} ({s['share_months_hit_above_60']:.0%})  {n['cum_return_pct_median']:+7.1f}/{n['cum_return_pct_p95']:+7.1f}   {n['max_drawdown_pct_median']:6.1f}/{n['max_drawdown_pct_p95']:6.1f}")
    out.append("")
fin = [json.loads(l) for l in open(os.path.join(os.path.dirname(HERE), "hourly/final.jsonl"))]
cl = [r for r in fin if r["tag"] == "T12_tp1.5_sl1.5_rich_hgb_barrier_mag"][0]["final"]["meta"]
no = json.load(open(f"{HERE}/c_h38_hourly_barrier_meta_nextopen.json"))["extra"]["final_meta_nextopen"]
out.append("== H38 hourly barrier, held-out meta selection (hourly.py --final fields)")
out.append("entry      n     hit     bps    null95   monthly bps 95% t-interval   lower>0  bps>null95")
for lab, r in (("close", cl), ("nextopen", no)):
    ci = r["monthly_bps_ci"]
    out.append(f"{lab:9s} {r['n']:5d}  {r['hit']:.4f}  {r['bps']:+6.2f}  {r['null95']:6.2f}   [{ci[0]:+7.2f}, {ci[1]:+7.2f}]   {str(ci[0] > 0):7s}  {r['bps'] > r['null95']}")
open(f"{HERE}/h58_side_by_side.txt", "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out))
