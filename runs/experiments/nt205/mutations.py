import subprocess,sys
PY="C:/Users/Step/miniforge3/envs/nt/python"
R="D:/nt/nt_wt_205/"
muts=[
("src/neural_trade/notebook/calibration_ui.py","if not self.matches_saved() or self.pipeline is None:\n            return False","return self.matches_saved()\n        if False:\n            return False"),
("src/neural_trade/notebook/calibration_ui.py","if callable(sig) and sig() != self.pipeline.direction_signal():","if False:"),
("src/neural_trade/notebook/calibration_ui.py","        return all(abs(float(mine.get(h, 1.0))","        return True or all(abs(float(mine.get(h, 1.0))"),
("src/neural_trade/notebook/calibration_ui.py","[\"saved\"] if p_saved is not None and callable(saved_sig)","[\"saved\"] if False and callable(saved_sig)"),
("src/neural_trade/notebook/calibration_ui.py","not self.reproduces_saved()\n        p_cal","not self.matches_saved()\n        p_cal"),
("src/neural_trade/serving/predictor.py","if self.direction_signal:\n                # D-066","if False:\n                # D-066"),
("src/neural_trade/serving/predictor.py","\"direction_signal\": str(sig.get(h, \"ok\")) if sig else \"n/a\"}","\"x\": 1}"),
("src/neural_trade/cli.py","    if none:  # D-066","    if False:  # D-066"),
("src/neural_trade/calibration/online_calibrator.py","float(np.clip(init.get(h, 1.0), T_min, T_max))","float(init.get(h, 1.0))"),
]
for f,a,b in muts:
    p=R+f; s=open(p,encoding='utf-8',newline='').read()
    a2=a.replace('\n','\r\n') if '\r\n' in s else a
    b2=b.replace('\n','\r\n') if '\r\n' in s else b
    if a2 not in s: print("NOT FOUND",a[:50]); continue
    open(p,'w',encoding='utf-8',newline='').write(s.replace(a2,b2,1))
    try:
        r=subprocess.run([PY,"-m","pytest","-q","-p","no:cacheprovider","tests/test_nt205_nt204.py","-x"],cwd=R,capture_output=True,text=True,env={**__import__('os').environ,"PYTHONPATH":R+"src","CUDA_VISIBLE_DEVICES":"-1"})
        print("KILLED" if r.returncode else "SURVIVED", a[:60].replace("\n"," "), "|", r.stdout.strip().splitlines()[-1])
    finally:
        open(p,'w',encoding='utf-8',newline='').write(s)
