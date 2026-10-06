import json,glob,statistics as st
res={}
for d in sorted(glob.glob("runs/*")):
    side=d.split("-")[-1][0]
    s=json.load(open(d+"/status.json"))
    t=[json.loads(l)["time"] for l in open(d+"/metrics.jsonl")]
    dt=[b-a for a,b in zip(t,t[1:])]
    e=st.median(dt)
    print(d.split("-")[-1], s["epochs_completed"], round(s["sec_per_step"],4), "med_epoch_s", round(e,2), "elapsed", round(s["elapsed_seconds"],1))
    res.setdefault(side,[]).append((s["sec_per_step"],e))
for k,i in (("sec_per_step",0),("epoch_s",1)):
    a=[x[i] for x in res["A"]]; b=[x[i] for x in res["B"]]
    ma,mb=st.median(a),st.median(b)
    print(k,"A med",round(ma,4),"range",round(min(a),4),round(max(a),4),"spread",round((max(a)-min(a))/ma,3),"| B med",round(mb,4),"range",round(min(b),4),round(max(b),4),"spread",round((max(b)-min(b))/mb,3),"| ratio",round(mb/ma,3),"disjoint",min(b)>max(a))
