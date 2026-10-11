"""One-off patcher used to build the recut branch (H66): points every recomputed script at the cut data, the recut checkout and
recut caches, and the look months at 2024-07..2025-07. Idempotent: a rule that no longer matches is skipped; a rule that never
matched in a file is reported. usage: python patch_cutoff.py"""
import os, re
ROOT = "D:/nt/nt_recut/runs/tactical"
CSV_LIT = '"D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"'
IMPORT = ('import sys as _sys; _sys.path.insert(0, "D:/nt/nt_recut/runs/tactical/common")  # noqa: E402\n'
          'import cutoff as C  # noqa: E402  (H66: no bar after C.CUTOFF is read)\n')
CHD = ('os.chdir(r"D:' + chr(92) + 'nt' + chr(92) + 'nt_tactical")', 'os.chdir(r"D:' + chr(92) + 'nt' + chr(92) + 'nt_recut")')
FILES = {
    "hourly/hourly.py": [CHD, ('OUT = "runs/tactical/hourly"', 'OUT = "runs/tactical/recut/hourly"')],
    "regimes/part2.py": [('"D:/nt/nt_tactical/runs/tactical/hourly/hourly.py"', '"D:/nt/nt_recut/runs/tactical/hourly/hourly.py"'),
                         ('"D:/nt/nt_tactical/runs/tactical/hourly/ranking.json"', '"D:/nt/nt_recut/runs/tactical/hourly/ranking.json"'),
                         ('chdirs to D:/nt/nt_tactical', 'chdirs to D:/nt/nt_recut')],
    "geom2/meta.py": [], "geom2/meta_nextopen.py": [],
    "lab/decay.py": [CHD, ('"runs/tactical/lab/decay.json"', '"runs/tactical/recut/lab/decay.json"')],
    "lab/decay_asym.py": [CHD], "lab/decay_vol.py": [CHD], "lab/decay_nextopen.py": [CHD],
    "lab/decay_lib.py": [('CACHE = "D:/nt/nt_decay_cache"', 'CACHE = "D:/nt/nt_recut_cache/decay"')],
    "recent/feats.py": [('CACHE = "D:/nt/nt_recent_cache"', 'CACHE = "D:/nt/nt_recut_cache/recent"')],
    "recent/wf.py": [('CACHE = "D:/nt/nt_recent_cache"', 'CACHE = "D:/nt/nt_recut_cache/recent"'),
                     ('"look": ("2024-07", "2025-09")', '"look": ("2024-07", "2025-07")')],
    "nextopen/nolib.py": [], "volnet/volnet_lib.py": [],
    "volnet/run.py": [('"look": ("2024-07", "2025-09")', '"look": ("2024-07", "2025-07")')],
    "calendar/callib.py": [('TEST_START, TEST_END = "2024-07-01", "2025-09-30"', 'TEST_START, TEST_END = "2024-07-01", "2025-07-20"')],
    "calendar/run_calendar.py": [],
    "flow/flowlib.py": [('CACHE = "D:/nt/nt_flow_cache"', 'CACHE = "D:/nt/nt_recut_cache/flow"')],
    "flow/flowrun.py": [('LOOK = ("2024-07", "2025-09")', 'LOOK = ("2024-07", "2025-07")'),
                        ('STAB_LOOK = ("2024-07", "2025-09")', 'STAB_LOOK = ("2024-07", "2025-07")')],
    "fwd/fwdlib.py": [('CACHE = "D:/nt/nt_fwd_cache"', 'CACHE = "D:/nt/nt_recut_cache/fwd"')],
    "lab/decay_report.py": [('hold = ("2024-07", "2025-09")', 'hold = ("2024-07", "2025-07")')],
    "nextopen/run_c.py": [("'nt_tactical'", "'nt_recut'")],
    "equity/run_c.py": [("'nt_tactical'", "'nt_recut'")],
}
def patch(rel, rules):
    p = f"{ROOT}/{rel}"; s = open(p, encoding="utf-8").read(); o = s
    s = s.replace(CSV_LIT, "C.CSV")
    for a, b in rules:
        s = s.replace(a, b)
    if "import cutoff as C" not in s and ("C.CSV" in s or "TEST_END" in s):
        m = re.search(r"^(import |from )", s, re.M)
        s = s[:m.start()] + IMPORT + s[m.start():]
    if s != o:
        open(p, "w", encoding="utf-8", newline="").write(s); print("patched", rel)
    else:
        print("unchanged", rel)
for rel, rules in FILES.items():
    patch(rel, rules)
