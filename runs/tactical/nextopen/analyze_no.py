"""H56 look table, close vs next-open side by side (analyze.look's own computation, run on each variant's rows).
usage: python analyze_no.py"""
import importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REC = os.path.join(os.path.dirname(HERE), "recent")
spec = importlib.util.spec_from_file_location("analyze_rec", f"{REC}/analyze.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
res = {}
for entry in ("close", "nextopen"):
    A.HERE = os.path.join(HERE, f"h56_{entry}")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        A.look()
    res[entry] = json.load(open(f"{A.HERE}/look.json"))
json.dump(res, open(f"{HERE}/h56_side_by_side.json", "w"), indent=1)
txt = []
for bar in ("1m", "1h"):
    txt.append(f"== H56 {bar} look 2024-07..2025-07-20 (recut): close entry vs next-open entry (the same picks; label return changes)")
    txt.append(f"{'pick':16s} {'entry':9s} {'AUC':>7s} {'base':>7s} {'diff95 [lo,hi]':>26s} ok95 ok98.3 {'top10 hit':>9s} {'bps':>7s}")
    for c, n in zip(res["close"][bar], res["nextopen"][bar]):
        for lab, r in (("close", c), ("nextopen", n)):
            d = r["diff95"]
            txt.append(f"{r['pick']:16s} {lab:9s} {r['auc'][0]:7.4f} {r['base_auc'][0]:7.4f} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] {str(r['success_95']):5s} {str(r['success_9833']):6s} {r['top10_hit']:9.4f} {r['top10_bps']:+7.2f}")
    txt.append("")
open(f"{HERE}/h56_side_by_side.txt", "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt))
