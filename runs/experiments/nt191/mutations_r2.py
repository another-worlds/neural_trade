"""NT-191 repair 2: apply each mutation to the worktree's stability.py, run the tests that must kill it, restore."""
import os
import subprocess
import sys
from pathlib import Path

WT = Path("D:/nt/nt_wt_191")
SRC = WT / "src/neural_trade/experiments/stability.py"
PY = "C:/Users/Step/miniforge3/envs/nt/python"
MUT = {
    "M6 the probe re-run's verdict replaces the first": (
        "            reruns += got\n",
        "            reruns += got\n            v.passed = all(g.passed for g in got) if got else v.passed\n",
        "probe_rerun_never_turns"),
    "M8c not_rerun not carried on retry": (
        "not_rerun: List[Verdict] = [Verdict.from_dict(d) for d in (origin or {}).get(\"not_rerun\", [])]",
        "not_rerun: List[Verdict] = []",
        "not_rerun_cells"),
    "M16 WinError lock back to a verdict": (
        "    if name == \"PermissionError\" and _winerror(message, winerror) in TRANSIENT_WINERRORS:\n        return True\n",
        "",
        "windows"),
    "M17 every PermissionError non-verdict": (
        "    if name == \"PermissionError\" and _winerror(message, winerror) in TRANSIENT_WINERRORS:",
        "    if name == \"PermissionError\":",
        "windows"),
    "M18 csv check removed": (
        "        if not resolve_data_path(csv).is_file():             # refused before any run directory exists\n"
        "            raise ValueError(f\"the bars file {csv} does not exist\")\n",
        "",
        "missing_bars"),
}
CRLF, LF = chr(13) + chr(10), chr(10)
orig = SRC.read_bytes()
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
try:
    for name, (a, b, k) in MUT.items():
        text = orig.decode("utf-8")
        if CRLF in text:
            a, b = a.replace(LF, CRLF), b.replace(LF, CRLF)
        assert text.count(a) == 1, name
        SRC.write_bytes(text.replace(a, b).encode("utf-8"))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow",
                            "tests/test_stability_harness.py", "-k", k], cwd=WT, env=env, capture_output=True, text=True)
        tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln or ln.startswith("FAILED")]
        print(f"{name}: rc {r.returncode} (killed={r.returncode != 0})", flush=True)
        for ln in tail:
            print("    ", ln, flush=True)
finally:
    SRC.write_bytes(orig)
    print("restored:", SRC.read_bytes() == orig)
sys.exit(0)
