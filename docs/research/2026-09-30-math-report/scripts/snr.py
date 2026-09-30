import glob, json, numpy as np, pandas as pd
B="D:/neural_trade/runs/scenarios/long_360d_stab"
L=lambda p: np.log(2.0/(np.asarray(p)-1.0))
out={}
for d in sorted(glob.glob(B+"/2026*")):
    tag=d.split("default__")[1]; init=json.load(open(d+"/period_init.json"))["configured"]
    h=pd.read_csv(d+"/indicator_params_history.csv").iloc[:-1]  # drop the restored row
    for k in init:
        l=np.concatenate([[L(init[k])],L(h[k].values)])
        dl=np.diff(l)
        out.setdefault(k,[]).append((abs(dl[0]), np.median(abs(dl[1:])), abs(l[-1]-l[0])/len(dl), np.sum(abs(dl))))
for k,v in out.items():
    v=np.array(v); print(f"{k:14s} ep0 |dl| {v[:,0].mean():.3f}  median later |dl|/epoch {v[:,1].mean():.3f}  net/epoch {v[:,2].mean():.3f}  path/net {np.mean(v[:,3]/np.maximum(v[:,2]*1,1e-9)/1):.1f}")
