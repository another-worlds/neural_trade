"""O11 part b prospective scoring, for a LATER session; NOT run in the O11 task. usage: python score_P_b.py [--csv path]
Loads frozen_P_b.json (refuses on a sha256 mismatch). Scores the first 3 complete calendar months after 2026-10 (2026-11, 2026-12, 2027-01) ONCE; refuses when
fewer than 3 are complete (before reading a bar) and refuses a second scoring. Criterion: o11_freeze.py docstring. A pass says only that the P criterion holds."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import o11_freeze as Z  # noqa: E402
if __name__ == "__main__":
    Z.main_score("b", os.path.join(HERE, "frozen_P_b.json"), os.path.join(HERE, "results_P_b.json"), Z.score_b_core, Z.T.FWD_START)
