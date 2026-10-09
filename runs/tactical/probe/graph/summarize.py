"""Markdown tables from gprobe_*.json: per-epoch gradient shares per term and group, the cosine matrix of the
leading terms, and the H16 eager numbers beside them.  usage: python summarize.py > summary.md"""
import glob, json
import numpy as np

H16 = json.load(open("runs/tactical/probe/probe_2021-10-13_s0_eager.json"))["probe"]
for f in sorted(glob.glob("runs/tactical/probe/graph/gprobe_*.json")):
    d = json.load(open(f)); P = d["probe"]
    terms = sorted({k[len("probe_grad_share_"):].rsplit("_", 1)[0] for k in P if k.startswith("probe_grad_share_")})
    groups = ("trunk", "head", "indicator")
    n_ep = len(next(iter(P.values())))
    print(f"\n## {d['data_end']} seed {d['seed']}  train {d['train_s']:.0f}s, {d['steps_per_epoch']} steps/epoch, probe every {d['probe_every']}\n")
    for g in groups:
        rows = [(t, [P[f"probe_grad_share_{t}_{g}"][e] for e in range(n_ep)]) for t in terms if f"probe_grad_share_{t}_{g}" in P]
        rows = [r for r in rows if max(r[1]) >= 0.01]
        if not rows:
            continue
        print(f"### {g}: share of gradient norm (rows with share >= 1% in some epoch)\n")
        print("| term | " + " | ".join(f"e{e+1}" for e in range(n_ep)) + " | H16 init | H16 trained |")
        print("|---|" + "---|" * (n_ep + 2))
        for t, v in sorted(rows, key=lambda r: -r[1][-1]):
            h = H16.get(f"probe_grad_share_{t}_{g}", [None, None])
            print(f"| {t} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {h[0]:.3f} | {h[1]:.3f} |" if h[0] is not None else f"| {t} | " + " | ".join(f"{x:.3f}" for x in v) + " | - | - |")
        print()
    top = [t for t, _ in sorted(((t, P[f"probe_grad_share_{t}_trunk"][-1]) for t in terms), key=lambda r: -r[1])[:6]]
    for g in groups:
        for e in (0, n_ep - 1):
            print(f"### cosine matrix, {g}, epoch {e+1} (six leading trunk terms)\n")
            print("| | " + " | ".join(top) + " |"); print("|---|" + "---|" * len(top))
            for a in top:
                cells = []
                for b in top:
                    if a == b: cells.append("1"); continue
                    k = f"probe_pcos_{a}__{b}_{g}" if f"probe_pcos_{a}__{b}_{g}" in P else f"probe_pcos_{b}__{a}_{g}"
                    cells.append(f"{P[k][e]:+.2f}" if k in P else "-")
                print(f"| {a} | " + " | ".join(cells) + " |")
            print()
