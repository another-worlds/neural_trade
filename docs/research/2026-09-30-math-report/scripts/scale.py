import pandas as pd, numpy as np, glob
B="D:/neural_trade/runs/scenarios/long_360d_stab"
h=pd.read_csv(glob.glob(B+"/*f-2__s0/indicator_params_history.csv")[0])
print("lr main first/last", h.log_lr_used.iloc[0], h.log_lr_used.min(), " ind", h.log_lr_indicator_used.min(), h.log_lr_indicator_used.max())
for d in sorted(glob.glob(B+"/2026*")):
    h=pd.read_csv(d+"/indicator_params_history.csv"); print(d[-8:], "main lr min", h.log_lr_used.min(), "ind lr min", h.log_lr_indicator_used.min())
z=np.load(glob.glob(B+"/*f-2__s0/predictions_oos.npz")[0])
t0=pd.to_datetime(z["extra__anchor_timestamp"]).min(); print("dev start", t0)
c=pd.read_csv("D:/neural_trade/Bitcoin_BTCUSDT.csv",usecols=["timestamp","close"],parse_dates=["timestamp"]).set_index("timestamp").close
s=c[(c.index>=t0-pd.Timedelta(days=430))&(c.index<t0)]
d10=(s.shift(-10)-s).dropna(); r10=(np.log(s.shift(-10))-np.log(s)).dropna()
m=pd.DataFrame({"usd":d10,"bps":r10*1e4,"px":s}).resample("M").agg({"usd":"std","bps":"std","px":"mean"})
print(m.round(1).to_string())
