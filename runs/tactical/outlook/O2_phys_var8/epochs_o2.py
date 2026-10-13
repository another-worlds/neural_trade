import glob, json, collections
tot = collections.Counter(); proc = 0
for c in ("on", "off"):
    ep = collections.Counter(); bad = []; w = 0
    for f in glob.glob(f"D:/nt/nt_tactical_outlook/runs/tactical/screens/o2_none_{c}/results*.jsonl"):
        for l in open(f, encoding="utf-8"):
            x = json.loads(l); ep[x["health"]["epochs_run"]] += 1; w += x["wall_s"]
            if not x["passed"] or not x["health"]["finite"] or x["health"]["nonfinite_grad_steps"]: bad.append((x["data_end"], x["seed"], x["reasons"]))
    print(c, "epochs_run counts", dict(ep), "summed run wall_s", round(w), "flagged", bad); proc += w
print("total summed process time h", round(proc / 3600, 2))
