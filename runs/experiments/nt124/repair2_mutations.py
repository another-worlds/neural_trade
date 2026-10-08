import os, subprocess
from pathlib import Path
ROOT = Path("D:/nt/nt_wt_124")
PY = "C:/Users/Step/miniforge3/envs/nt/python"
MUTS = {
    "M9 saved drops signal": ("src/neural_trade/notebook/calibration_ui.py",
                              "SAVED_ALPHA, sig() if callable(sig) else None)", "SAVED_ALPHA, None)"),
    "R2a transpose (object dtypes)": ("src/neural_trade/notebook/calibration_ui.py",
                                      'pd.DataFrame.from_dict(rows, orient="index")', "pd.DataFrame(rows).T"),
    "R2b no infer_objects": ("src/neural_trade/notebook/calibration_ui.py",
                             'names=["horizon", "pipeline"])).infer_objects()', 'names=["horizon", "pipeline"]))'),
    "R2c no na_rep": ("src/neural_trade/notebook/calibration_ui.py", ', na_rep="n/a")', ")"),
    "R2d widget unstyled": ("src/neural_trade/notebook/calibration_ui.py",
                            "show(table, self.comparison_table(styled=True))", "show(table, self.comparison_table().round(4))"),
    "R2e ece_na ignored": ("src/neural_trade/visualization/calibration_plots.py",
                           'if s in ece_na else', 'if False else'),
}
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}
for name, (rel, old, new) in MUTS.items():
    p = ROOT / rel
    src = p.read_text(encoding="utf-8")
    assert src.count(old) == 1, (name, src.count(old))
    p.write_text(src.replace(old, new), encoding="utf-8")
    try:
        t = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_direction_signal_surfaces.py"],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = [ln for ln in t.stdout.splitlines() if ln.startswith("FAILED") or " passed" in ln or " failed" in ln]
        print(f"{name}: {lines}", flush=True)
    finally:
        p.write_text(src, encoding="utf-8")
