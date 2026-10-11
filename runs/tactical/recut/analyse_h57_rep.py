"""Second run of the H57 look on the cut data (GPU nondeterminism): same paired analysis as volnet/analyse.py on results_look_recut_rep_0.jsonl."""
import json, sys
sys.path.insert(0, "D:/nt/nt_recut/runs/tactical/volnet")
import pandas as pd, analyse as A
df = pd.DataFrame([json.loads(l) for l in open("D:/nt/nt_recut/runs/tactical/volnet/results_look_recut_rep_0.jsonl")])
out, txt = A.table(df, ["ens"], "H57 look, recut, SECOND run (same code, same data, GPU nondeterminism)")
v = {str(H): bool(out[f"ens_{H}"]["sp"]["lo"] > 0 and out[f"ens_{H}"]["ql"]["lo"] > 0) for H in sorted(df.H.unique())}
open("D:/nt/nt_recut/runs/tactical/volnet/look_tables_recut_rep.txt", "w").write(txt + f"\n\nsuccess by horizon: {v}\n"); print(txt); print(v)
