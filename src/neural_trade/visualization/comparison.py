"""Figures for comparing runs (experiments.compare.compare_runs) and ablation verdicts.

``runs_comparison_figure`` draws one panel per metric kind (``direction/auc``, ``variance/coverage90``
...), the runs down the side (each label written once, one row per run, so two runs can never be
merged into one bar), and one dot per run and horizon in the horizon's colour, with:

* the no-skill / target reference of the metric (AUC 0.5, MCC 0, EV 0, CRPSS 0, coverage90 0.90 ...),
  dashed, and the axis zoomed around it and the data (never forced to start at 0);
* a 95% interval where one has a closed form, on effective samples N / horizon bars as in the
  evaluation report (direction metrics count only the samples outside the deadband). The sample
  sizes come from each run's evaluation report (``run_dirs=``, the glob given to compare_runs; else
  ``df.attrs['run_dirs']``; else the run ids looked up under the nearest ``runs/`` folder) or from
  ``{h}/n``, ``{h}/n_eff``, ``{h}/direction/n_masked`` columns; an interval stored in the report
  (``<metric>_ci``) is used as it is;
* the best baseline of the run's own report (e.g. logreg_lags) as a grey open diamond; none where
  that baseline sits on the reference (e.g. class_prior at AUC 0.5), which the subtitle counts;
* "n/a" where a run has no value, "0 (β=0)" where calibration shrank the served delta to zero
  (written beside the dot, on the side away from the reference line), and a note on a panel no run
  has (e.g. backtest/* when no backtest was scored);
* whether the runs were scored on one block: runs whose reports agree on the sample counts, the
  realised up-rates and the zero-delta baseline's RMSE share it (their intervals overlap by
  construction); otherwise each block gets a letter, shown after the run label.
"""
from __future__ import annotations

import glob as _glob
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.theme import apply

VERDICT_COLORS = {"VALUE": "#0ca30c", "HARMFUL": "#d03b3b", "NEUTRAL": "#898781", "INCONCLUSIVE": "#c98500"}

# no-skill / target reference per metric name (the last path component)
REFERENCE = {"auc": (0.5, "chance"), "bal_acc": (0.5, "chance"), "acc": (0.5, "coin flip"),
             "mcc": (0.0, "no skill"), "ev": (0.0, "zero delta"), "corr": (0.0, "no relation"),
             "skill_vs_zero": (0.0, "zero delta"), "crpss": (0.0, "const-variance baseline"),
             "corr_var_err2_spearman": (0.0, "no relation"), "coverage90": (0.9, "target"),
             "sharpe_net": (0.0, "zero"), "sharpe": (0.0, "zero"), "pred_up_rate": (0.5, "even split")}
# smallest half-span of the value axis around the reference, so noise cannot fill a panel
MIN_HALF_SPAN = {"auc": 0.03, "bal_acc": 0.03, "acc": 0.03, "mcc": 0.05, "ev": 0.005, "corr": 0.05,
                 "skill_vs_zero": 0.005, "crpss": 0.01, "corr_var_err2_spearman": 0.05, "coverage90": 0.04,
                 "sharpe_net": 1.0, "sharpe": 1.0}
CLOSER_TO_TARGET = {"coverage90": "closer to 0.90 is better", "pred_up_rate": "closer to the up-rate is better"}
_RUN_ID = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z-([0-9a-f]{7,40})(-dirty)?-([0-9a-f]{8})(?:-(.+))?$")


# ------------------------------------------------------------------ labels
def run_labels(run_ids: Sequence[str]) -> List[str]:
    """Short unique labels for run ids ``<YYYYMMDD>T<HHMMSS>Z-<sha>[-dirty]-<cfghash>[-name]``:
    'MM-DD HH:MM sha[*] [name]' (* = dirty tree), adding the seconds and then the config hash only
    when needed to keep them unique; ids that do not parse are kept whole."""
    ids = [str(i) for i in run_ids]
    parsed = [_RUN_ID.match(i) for i in ids]
    years = {m.group(1) for m in parsed if m}
    cfgs = {m.group(9) for m in parsed if m}

    def fmt(m, level):
        yr, mo, d, hh, mm, ss, sha, dirty, cfg, name = m.groups()
        s = (f"{yr}-" if len(years) > 1 else "") + f"{mo}-{d} {hh}:{mm}" + (f":{ss}" if level >= 1 else "")
        s += f" {sha[:7]}{'*' if dirty else ''}"
        if len(cfgs) > 1 or level >= 2:
            s += f" {cfg}"
        return s + (f" {name}" if name else "")

    for level in range(3):
        labels = [fmt(m, level) if m else i for m, i in zip(parsed, ids)]
        if len(set(labels)) == len(labels):
            return labels
    seen: Dict[str, int] = {}
    out = []
    for i in ids:                                  # the same run id twice: number the repeats
        seen[i] = seen.get(i, 0) + 1
        out.append(i if seen[i] == 1 else f"{i} #{seen[i]}")
    return out


# ------------------------------------------------------------------ run reports
# where a run directory can sit below a runs/ folder: runs/<id>, runs/<group>/<name>/runs/<id>
# (ablations, experiments) and runs/<name>/runs/<id>
_RUN_PATTERNS = ("{id}", "*/*/runs/{id}", "*/runs/{id}")


def find_run_dirs(run_ids: Sequence[str], start: Optional[Path] = None, levels: int = 3) -> Tuple[Dict[str, Path], Optional[Path]]:
    """Look the run ids up under the ``runs/`` folder of ``start`` (the working directory) or of one
    of its first ``levels`` parents (a notebook runs from notebooks/). Returns ({run id: directory},
    the runs folder searched), or ({}, None) when there is no runs folder."""
    start = Path.cwd() if start is None else Path(start)
    for base in [start, *list(start.parents)[:levels]]:
        root = base / "runs"
        if not root.is_dir():
            continue
        found: Dict[str, Path] = {}
        for rid in run_ids:
            for pat in _RUN_PATTERNS:
                hit = next(iter(sorted(root.glob(pat.format(id=_glob.escape(str(rid)))))), None)
                if hit is not None and hit.is_dir():
                    found[str(rid)] = hit
                    break
        return found, root
    return {}, None


def _load_reports(df, run_dirs, split: str) -> Tuple[Dict[str, dict], str]:
    """(run id -> {"report": eval report, "meta": calibration pipeline meta}, where the directories
    came from) for the runs in ``df``: ``run_dirs``, else ``df.attrs['run_dirs']``, else the run ids
    looked up under the nearest runs/ folder."""
    source = "given"
    if run_dirs is None:
        run_dirs, source = df.attrs.get("run_dirs"), "attrs"
    if run_dirs is None:
        found, root = find_run_dirs([str(i) for i in df.index])
        if not found:
            return {}, ""
        run_dirs, source = list(found.values()), f"looked up by run id under {root}"
    from neural_trade.experiments.compare import _dirs

    if isinstance(run_dirs, dict):
        run_dirs = list(run_dirs.values())
    out: Dict[str, dict] = {}
    for d in _dirs(run_dirs):
        p = d / f"eval_report_{split}.json"
        if not p.exists():
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        rid = rep.get("run_id") or d.name
        if rid not in df.index and d.name in df.index:
            rid = d.name
        if rid in df.index and rid not in out:
            mp = Path(d) / "artifacts" / "calibration" / "pipeline_meta.json"
            out[rid] = {"report": rep, "meta": json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}}
    return out, source


def _rounded(v, digits: int = 8):
    return round(float(v), digits) if isinstance(v, (int, float)) and v is not None else None


def block_key(report: Optional[dict]) -> Optional[tuple]:
    """What identifies the block a report was scored on: its sample counts and deadband, and per
    horizon the labelled count, the realised up-rate and the zero-delta baseline's RMSE (functions of
    the realised outcomes only). Two runs on the same block give equal keys; None without a report."""
    if not report:
        return None
    key: list = [report.get("n"), _rounded(report.get("deadband_bps"))]
    zero = ((report.get("baselines") or {}).get("zero_delta") or {}).get("horizons") or {}
    for h, row in sorted(((report.get("model") or {}).get("horizons") or {}).items()):
        d = row.get("direction") or {}
        key += [h, row.get("n"), d.get("n_masked"), _rounded(d.get("true_up_rate")),
                _rounded(((zero.get(h) or {}).get("delta") or {}).get("rmse"), 4)]
    return tuple(key)


def _split_metric(m: str):
    """'h1/direction/auc' -> ('h1', 'direction', 'auc'); 'backtest/sharpe_net' -> (None, 'backtest', 'sharpe_net')."""
    parts = m.split("/")
    if parts[0] in T.HORIZONS and len(parts) >= 3:
        return parts[0], parts[1], "/".join(parts[2:])
    return None, parts[0] if len(parts) > 1 else "", parts[-1]


class _RunFacts:
    """What a metric's interval, baseline and zero-label need about one run: sample sizes, horizon
    steps, baselines and the delta scale, from its report or from ``df`` columns."""

    def __init__(self, df, reports):
        self.df, self.reports = df, reports

    def _col(self, rid, key):
        if key in self.df.columns:
            v = pd.to_numeric(pd.Series([self.df.at[rid, key]]), errors="coerce").iloc[0]
            return None if pd.isna(v) else float(v)
        return None

    def row(self, rid, h):
        rep = self.reports.get(rid, {}).get("report") or {}
        return (rep.get("model", {}).get("horizons", {}) or {}).get(h) or {}

    def value(self, rid, h, group, key):
        v = (self.row(rid, h).get(group) or {}).get(key)
        return float(v) if isinstance(v, (int, float)) and v is not None else self._col(rid, f"{h}/{group}/{key}")

    def n(self, rid, h):
        v = self.row(rid, h).get("n")
        return float(v) if v is not None else self._col(rid, f"{h}/n")

    def steps(self, rid, h) -> Optional[int]:
        rep = self.reports.get(rid, {}).get("report") or {}
        hs = (rep.get("meta") or {}).get("horizon_steps")
        if hs is not None:
            return int(hs[T.HORIZONS.index(h)])
        n, ne = self.n(rid, h), (self.row(rid, h).get("n_eff") or self._col(rid, f"{h}/n_eff"))
        return int(round(n / ne)) if n and ne else None

    def beta(self, rid, h) -> Optional[float]:
        b = ((self.reports.get(rid, {}).get("meta") or {}).get("delta_scale") or {}).get(h)
        return float(b) if b is not None else self._col(rid, f"{h}/delta/beta")

    def baseline(self, rid, h, group, key, higher: Optional[bool]):
        """(name, value) of the best baseline in the run's report for this metric, or None."""
        if higher is None:
            return None
        rep = self.reports.get(rid, {}).get("report") or {}
        best = None
        for name, b in (rep.get("baselines") or {}).items():
            v = (((b.get("horizons") or {}).get(h) or {}).get(group) or {}).get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                if best is None or (v > best[1] if higher else v < best[1]):
                    best = (name, float(v))
        return best

    def interval(self, rid, h, group, key, v):
        """95% (lo, hi) for the value ``v``, or None when there is no closed form or no sample size."""
        ci = (self.row(rid, h).get(group) or {}).get(f"{key}_ci")
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(c, (int, float)) for c in ci):
            return float(ci[0]), float(ci[1])
        steps = self.steps(rid, h)
        if steps is None or not np.isfinite(v):
            return None
        if group in ("direction", "gauss_direction"):
            nd = self.value(rid, h, group, "n_masked")
            if not nd:
                return None
            if key == "auc":
                up = self.value(rid, h, group, "true_up_rate")
                if up is None:
                    return None
                npos = max(1.0, round(nd * up))
                return S.auc_ci(v, npos, max(1.0, nd - npos), steps=steps)
            if key == "mcc":
                half = S.corr_null(nd, steps=steps) * (1 - v * v)
                return v - half, v + half
            if key in ("acc", "bal_acc"):
                _, lo, hi = S.wilson(v * nd, nd, steps=steps)
                return float(lo), float(hi)
            return None
        n = self.n(rid, h)
        if not n:
            return None
        if group == "delta" and key == "corr":
            half = S.corr_null(n, steps=steps) * (1 - v * v)
            return v - half, v + half
        if group == "variance" and key == "coverage90":
            _, lo, hi = S.wilson(v * n, n, steps=steps)
            return float(lo), float(hi)
        return None


def _better(key: str) -> Optional[bool]:
    from neural_trade.evaluation.report import HIGHER_IS_BETTER, LOWER_IS_BETTER

    if key in HIGHER_IS_BETTER or key.startswith("sharpe") or key in ("total_return", "hit_rate"):
        return True
    if key in LOWER_IS_BETTER or key in ("width90", "max_drawdown", "turnover"):
        return False
    return None


# ------------------------------------------------------------------ the figure
# rough text geometry for keeping labels inside their panel: px per character of the 10-11 px
# labels, and the narrowest figure the layout is sized for (a notebook cell)
_CHAR_PX, _MIN_FIG_PX = 6.2, 900
_DOT_GAP = "  "                       # an en space and a space: text starts clear of a 9 px dot
_SUB_CHARS = 140                           # subtitle lines wrap here, so they fit a 900 px figure


def _pads(a: float, b: float, right: List[Tuple[float, int]], left: List[Tuple[float, int]], panel_px: float,
          base: float = 0.08) -> Tuple[float, float]:
    """(left, right) axis padding as fractions of ``b - a`` that leave room for the text written
    beside the dots: ``right`` / ``left`` hold (x where a text starts, its number of characters)."""
    w = (b - a) or 1.0
    pl = pr = base
    for _ in range(4):                     # the room a text needs grows with the padding itself
        total = 1.0 + pl + pr
        need_r = max([nc * _CHAR_PX / panel_px * total - (b - x) / w for x, nc in right] + [0.0])
        need_l = max([nc * _CHAR_PX / panel_px * total - (x - a) / w for x, nc in left] + [0.0])
        pr, pl = min(1.5, max(base, need_r + 0.02)), min(1.5, max(base, need_l + 0.02))
    return pl, pr


def _text_start(diamond: Optional[float], x: float, right: bool, on_mark: bool) -> Tuple[float, str]:
    """(x where a text written beside a dot starts, the gap before it): past ``x`` (the dot or its
    interval's cap), and past the row's baseline diamond when that lies further out on the same side.
    The gap is wider when the text starts at a marker, so the marker does not cover it."""
    if diamond is not None and ((right and diamond >= x) or (not right and diamond <= x)):
        return float(diamond), _DOT_GAP
    return float(x), (_DOT_GAP if on_mark else " ")


def _block_letters(ids: Sequence[str], keys: Dict[str, Optional[tuple]]) -> Dict[tuple, str]:
    letters: Dict[tuple, str] = {}
    for rid in ids:
        k = keys[rid]
        if k is not None and k not in letters:
            n = len(letters)
            letters[k] = chr(ord("A") + n) if n < 26 else str(n + 1)
    return letters


def _block_lines(ids, keys, letters, reports, split: str, h_show: str) -> List[str]:
    """Subtitle lines on the blocks the runs were scored on."""
    missing = sum(1 for rid in ids if keys[rid] is None)
    miss_txt = f" ({missing} run{'s' if missing != 1 else ''} without a report: block unknown)" if missing else ""
    if len(letters) == 1:
        return [f"the runs share one {split} block (equal sample counts, up-rates and zero-delta RMSE){miss_txt}; "
                "their intervals overlap by construction, so a difference between two runs needs a paired test"]
    parts = []
    for k, letter in letters.items():
        rid = next(r for r in ids if keys[r] == k)
        rep = reports[rid]["report"]
        row = ((rep.get("model") or {}).get("horizons") or {}).get(h_show) or {}
        up = (row.get("direction") or {}).get("true_up_rate")
        zero = ((((rep.get("baselines") or {}).get("zero_delta") or {}).get("horizons") or {}).get(h_show) or {})
        rmse = (zero.get("delta") or {}).get("rmse")
        count = sum(1 for r in ids if keys[r] == k)
        desc = [f"n {rep.get('n'):,}" if isinstance(rep.get("n"), int) else None,
                f"{h_show} up-rate {up:.3f}" if isinstance(up, (int, float)) else None,
                f"zero-delta RMSE ${rmse:,.0f}" if isinstance(rmse, (int, float)) else None]
        parts.append(f"[{letter}] " + ", ".join(d for d in desc if d) + f" ({count} run{'s' if count != 1 else ''})")
    return [f"the runs come from {len(letters)} different {split} blocks (letter after each run label){miss_txt}; "
            + "; ".join(parts),
            "runs on one block have overlapping intervals by construction (compare them with a paired test); "
            "across blocks, a difference also carries the change of period"]


def _wrap(line: str, width: int = _SUB_CHARS) -> List[str]:
    """Wrap one subtitle line to ``width`` characters, breaking after its '; ' clauses where it can
    and at spaces inside a clause that is longer than a line."""
    out: List[str] = []
    cur = ""
    for seg in line.split("; "):
        cand = f"{cur}; {seg}" if cur else seg
        if len(cand) <= width:
            cur = cand
            continue
        if cur:
            out.append(cur + ";")
        parts = [seg] if len(seg) <= width else (
            textwrap.wrap(seg, width, break_long_words=False, break_on_hyphens=False) or [""])
        out += parts[:-1]
        cur = parts[-1]
    out.append(cur)
    return out


def _baseline_lines(baselines: Dict[str, Dict[Optional[str], dict]]) -> List[str]:
    """Subtitle lines naming the diamonds (best baseline per metric kind and horizon, with its value)
    and saying where no diamond is drawn because the best baseline sits on the reference line."""
    lines, flat = [], []
    for kind, per_h in baselines.items():
        parts, flat_h = [], []
        for h, rec in per_h.items():
            ticks, at = rec["ticks"], rec["at_ref"]
            hp = f"{h} " if h else ""
            if not ticks:
                flat_h.append(h or "")
                continue
            names = sorted({nm for nm, _ in ticks})
            vs = [v for _, v in ticks]
            val = f" {vs[0]:.3g}" if max(vs) - min(vs) < 5e-4 else f" {min(vs):.3g} to {max(vs):.3g} (by run)"
            part = f"{hp}{', '.join(names)}{val}"
            if at:
                part += f" in {len(ticks)} runs; {hp}none in the other {len(at)} (best baseline on the reference)"
            parts.append(part)
        if parts:
            if flat_h:
                parts.append(f"{', '.join(flat_h)} on the reference in every run: no diamond")
            lines.append(f"diamonds, {kind}: best baseline of each run's own report - " + "; ".join(parts))
        else:
            flat.append(kind)
    if flat:
        lines.append(f"no diamond for {', '.join(flat)}: the best baseline sits on the reference line in every run")
    return lines


def runs_comparison_figure(df, metrics: Optional[Sequence[str]] = None, *, title: Optional[str] = None,
                           run_dirs=None, split: Optional[str] = None, height: Optional[int] = None):
    """Runs side by side, one panel per metric kind (``df`` from compare_runs).

    ``run_dirs``: the run directories / glob given to compare_runs (else ``df.attrs['run_dirs']``,
    else the run ids are looked up under the nearest ``runs/`` folder); their evaluation reports
    supply the sample sizes for the 95% intervals, the baselines, the delta scale and which block
    each run was scored on. Without them the dots are drawn with no interval and the subtitle says so.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    split = split or df.attrs.get("split", "test")
    metrics = [m for m in (metrics or [c for c in df.columns if c not in ("seed", "tags")]) if m in df.columns]
    ids = [str(i) for i in df.index]
    n_runs = len(ids)
    reports, source = _load_reports(df, run_dirs, split)
    facts = _RunFacts(df, reports)
    keys = {rid: block_key((reports.get(rid) or {}).get("report")) for rid in ids}
    letters = _block_letters(ids, keys)
    labels = run_labels(ids)
    if len(letters) > 1:                   # several blocks: say which one each run was scored on
        labels = [f"{lab} [{letters[keys[rid]]}]" if keys[rid] is not None else f"{lab} [?]"
                  for lab, rid in zip(labels, ids)]

    kinds: Dict[str, List[str]] = {}
    for m in metrics:
        h, group, key = _split_metric(m)
        kinds.setdefault(f"{group}/{key}" if group else key, []).append(m)
    horizons = [h for h in T.HORIZONS if any(_split_metric(m)[0] == h for m in metrics)]
    grouped = len(horizons) > 1
    offsets = {h: o for h, o in zip(horizons, np.linspace(-0.22, 0.22, len(horizons)) if grouped else [0.0])}
    offsets[None] = 0.0

    def steps_label(h):
        st = next((facts.steps(r, h) for r in ids if facts.steps(r, h)), None)
        return f"{h} ({st} bars)" if st else h

    cols = min(3, max(1, len(kinds)))
    rows = int(np.ceil(len(kinds) / cols)) if kinds else 1
    row_px = max(150, 30 * n_runs * max(1, len(horizons)) ** 0.5 + 70)
    label_px = 1.1 * _CHAR_PX * max((len(lab) for lab in labels), default=10)
    panel_px = max(120.0, (_MIN_FIG_PX - label_px - 30) * (1 - 0.05 * (cols - 1)) / cols)
    titles = []
    for kind in kinds:
        key = kind.split("/")[-1]
        better = CLOSER_TO_TARGET.get(key) or {True: "higher is better", False: "lower is better"}.get(_better(key), "")
        ref = REFERENCE.get(key)
        second = " · ".join(s for s in (better, f"dashed: {ref[1]} {ref[0]:g}" if ref else "") if s)
        titles.append(f"<b>{kind}</b>" + (f"<br><span style='font-size:11px;color:{T.MUTED}'>{second}</span>"
                                          if second else ""))
    specs = ([[{} if r * cols + c < len(kinds) else None for c in range(cols)] for r in range(rows)]
             if kinds else [[{}]])
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles or None, shared_yaxes=True, specs=specs,
                        horizontal_spacing=0.05, vertical_spacing=min(0.3, 100 / (rows * row_px + 1)))
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=T.INK_2))
    seen = T.legend_once()
    no_ci, beta0_runs = set(), set()
    any_ci = False
    baselines: Dict[str, Dict[Optional[str], dict]] = {}
    for k, (kind, ms) in enumerate(kinds.items()):
        r, c = k // cols + 1, k % cols + 1
        key = kind.split("/")[-1]
        ref = REFERENCE.get(key)
        higher = _better(key)
        span = [ref[0]] if ref else []
        n_text = sum(1 for m in ms for v in pd.to_numeric(df[m], errors="coerce") if np.isfinite(v))
        show_values = n_text <= 10
        right_txt: List[Tuple[float, int]] = []
        left_txt: List[Tuple[float, int]] = []
        drawn = False
        for m in ms:
            h, group, _ = _split_metric(m)
            color = T.HORIZON_COLORS.get(h, T.INK_2)
            vals = pd.to_numeric(df[m], errors="coerce").to_numpy(float)
            if not np.isfinite(vals).any():
                continue
            drawn = True
            ypos = np.arange(n_runs) + offsets.get(h, 0.0)
            lo, hi, text, hover = np.full(n_runs, np.nan), np.full(n_runs, np.nan), [], []
            for j, rid in enumerate(ids):
                v = vals[j]
                ci = facts.interval(rid, h, group, key, v) if h and np.isfinite(v) else None
                if ci is not None:
                    lo[j], hi[j] = ci
                elif np.isfinite(v):
                    no_ci.add(kind)
                # an exact 0 of a delta metric means a constant served delta: say why, when the run says
                zero = group == "delta" and np.isfinite(v) and v == 0.0
                beta = facts.beta(rid, h) if zero and h else None
                if zero and beta == 0.0:
                    beta0_runs.add(labels[j])
                    why, tag = "served delta shrunk to 0 by calibration (β=0)", "0 (β=0)"
                elif zero:
                    why, tag = "exactly 0: the served delta is constant", "0 (constant δ)"
                else:
                    why, tag = "", (f"{v:.4g}" if show_values and np.isfinite(v) else "")
                text.append(tag)
                blk = f"<br>{split} block {letters[keys[rid]]}" if len(letters) > 1 and keys[rid] is not None else ""
                hover.append(f"{rid}<br>{m} = {v:.4g}" + (f"<br>95% interval {ci[0]:.4g} to {ci[1]:.4g}" if ci else "")
                             + (f"<br>{why}" if why else "") + blk)
            span += [x for x in np.r_[vals, lo, hi] if np.isfinite(x)]
            has_ci = np.isfinite(lo) & np.isfinite(vals)
            any_ci = any_ci or bool(has_ci.any())
            name = steps_label(h) if h else "value"
            fig.add_trace(go.Scatter(
                x=vals.astype(np.float32), y=ypos, mode="markers",
                marker=dict(size=9, color=color, line=dict(color=T.PAPER, width=1)),
                error_x=dict(type="data", symmetric=False, array=np.where(has_ci, hi - vals, 0).astype(np.float32),
                             arrayminus=np.where(has_ci, vals - lo, 0).astype(np.float32), color=color,
                             thickness=1.4, width=4),
                name=name, legendgroup=name, showlegend=seen(name), legendrank=1 + T.HORIZONS.index(h) if h else 4,
                hovertext=hover, hoverinfo="text"), r, c)
            # best baseline of each run's own report; none where it sits on the reference (e.g. class_prior)
            bl = [facts.baseline(rid, h, group, key, higher) if h else None for rid in ids]
            at_ref = [j for j, b in enumerate(bl) if b is not None and ref and abs(b[1] - ref[0]) <= 1e-9]
            bj = [j for j, b in enumerate(bl) if b is not None and j not in at_ref]
            diamond = {j: bl[j][1] for j in bj}

            # the value (or why it is exactly 0) beside the dot, past the end of its interval, on the side
            # away from the reference line: never on a neighbouring horizon's dot or on the dashed line
            centre = ref[0] if ref else float(np.nanmedian(vals))
            tj = [j for j in range(n_runs) if text[j]]
            if tj:
                xs, pos, txt = [], [], []
                for j in tj:
                    right = bool(vals[j] >= centre)
                    end = (hi[j] if right else lo[j]) if has_ci[j] else vals[j]
                    end, gap = _text_start(diamond.get(j), end, right, not has_ci[j])   # past the cap or a marker
                    xs.append(end)
                    pos.append("middle right" if right else "middle left")
                    txt.append(f"{gap}{text[j]}" if right else f"{text[j]}{gap}")
                    (right_txt if right else left_txt).append((float(end), len(text[j]) + len(gap)))
                fig.add_trace(go.Scatter(x=np.array(xs, np.float32), y=ypos[tj], mode="text", text=txt,
                                         textposition=pos, textfont=dict(size=10, color=color), cliponaxis=False,
                                         showlegend=False, hoverinfo="skip"), r, c)
            # missing values: say so on the row (beside the reference line, not on it), so a gap never
            # reads as a value
            miss = [j for j in range(n_runs) if not np.isfinite(vals[j])]
            if miss:
                x_na = ref[0] if ref else float(np.nanmedian(vals))
                na = [_text_start(diamond.get(j), x_na, True, False) for j in miss]
                right_txt += [(float(x), 3 + len(g)) for x, g in na]
                fig.add_trace(go.Scatter(x=[x for x, _ in na], y=ypos[miss], mode="text", text=[f"{g}n/a" for _, g in na],
                                         textposition="middle right", showlegend=False, cliponaxis=False,
                                         textfont=dict(size=10, color=color if grouped else T.MUTED),
                                         hoverinfo="skip"), r, c)
            if bj or at_ref:
                rec = baselines.setdefault(kind, {}).setdefault(h, {"ticks": [], "at_ref": []})
                rec["ticks"] += [bl[j] for j in bj]
                rec["at_ref"] += [bl[j][0] for j in at_ref]
            if bj:
                bx = np.array([bl[j][1] for j in bj], float)
                span += list(bx)
                fig.add_trace(go.Scatter(
                    x=bx.astype(np.float32), y=ypos[bj], mode="markers", name="best baseline (own report)",
                    legendgroup="baseline", showlegend=seen("best baseline"), legendrank=10,
                    marker=dict(symbol="diamond-open", size=10, color=T.NEUTRAL, line=dict(color=T.NEUTRAL, width=1.6)),
                    hovertext=[f"{ids[j]}<br>best baseline for {m}: {bl[j][0]} {bl[j][1]:.4g}" for j in bj],
                    hoverinfo="text"), r, c)
        if not drawn:
            continue
        if ref:
            fig.add_vline(x=ref[0], line=dict(color=T.NEUTRAL, dash="dash", width=1), row=r, col=c)
        span = np.array(span, float)
        a, b = float(span.min()), float(span.max())
        if ref and key in MIN_HALF_SPAN:
            a, b = min(a, ref[0] - MIN_HALF_SPAN[key]), max(b, ref[0] + MIN_HALF_SPAN[key])
        if b - a <= 0:
            a, b = a - 0.5 * (abs(a) or 1.0), b + 0.5 * (abs(b) or 1.0)
        pl, pr = _pads(a, b, right_txt, left_txt, panel_px)
        fig.update_xaxes(range=[a - pl * (b - a), b + pr * (b - a)], row=r, col=c)

    fig.update_xaxes(tickformat=".3~r", exponentformat="none", zeroline=False)
    fig.update_yaxes(tickmode="array", tickvals=list(range(n_runs)), ticktext=labels, range=[n_runs - 0.5, -0.5],
                     showgrid=False, zeroline=False, tickfont=dict(size=11, color=T.INK_2))
    n_ann = len(fig.layout.annotations)
    T.note_on_empty(fig, "not in these runs' eval reports<br>(backtest/* is scored only by<br>walk-forward and ablation runs)")
    for a in fig.layout.annotations[n_ann:]:
        a.update(xanchor="center", yanchor="middle")      # paper x > 2/3 would otherwise right-align it

    parsed = [_RUN_ID.match(i) for i in ids]
    lines = [f"{n_runs} runs scored on the {split} block (each run's eval_report_{split}.json); one row per run"
             + ("; row label: run start MM-DD HH:MM UTC and commit" if any(parsed) else "")
             + ("; * after the commit = run from a working tree with uncommitted changes"
                if any(m and m.group(8) for m in parsed) else "") + "; full run id on hover"]
    if any_ci:
        ci_txt = "bars: 95% interval on effective samples, N / horizon bars (direction: only samples outside the deadband)"
        if no_ci:
            ci_txt += f"; no interval for {', '.join(sorted(no_ci))} (no closed form)"
        lines.append(ci_txt)
    if reports:
        if source.startswith("looked up"):
            lines.append(f"reports {source} (pass run_dirs= to choose the directories)")
        lines += _block_lines(ids, keys, letters, reports, split, horizons[0] if horizons else "h1")
    else:
        lines.append("no evaluation reports found: " + ("" if any_ci else "no intervals, ")
                     + "no baselines or β labels, and whether the runs share a block is unknown; pass run_dirs= "
                     "(the glob given to compare_runs)")
    if beta0_runs:
        lines.append(f"β=0: calibration shrank the served delta to zero ({', '.join(sorted(beta0_runs))})")
    lines += _baseline_lines(baselines)
    same = _identical_runs(df, metrics, labels)
    if same:
        lines.append("identical metrics: " + "; ".join(" = ".join(g) for g in same))
    wrapped = [w for line in lines for w in _wrap(line)]
    sub = "<br>".join(wrapped)
    n_lines = len(wrapped)
    top = 124 + 20 * n_lines
    total = height or int(rows * row_px + top + 60)
    apply(fig, title=title or f"Run comparison: {n_runs} runs", subtitle=sub, height=total)
    plot_h = max(total - top - 48, 1)
    fig.update_layout(margin=dict(t=top, l=10), legend=dict(y=1.0 + 52 / plot_h))
    return fig


def _identical_runs(df, metrics, labels) -> List[List[str]]:
    """Groups of runs whose values are equal on every shown metric (e.g. a re-run of the same model)."""
    if len(df) < 2 or not metrics:
        return []
    vals = df[list(metrics)].apply(pd.to_numeric, errors="coerce")
    if vals.isna().all().all():
        return []
    keys = [tuple(np.round(row.to_numpy(float), 10)) for _, row in vals.iterrows()]
    groups: Dict[tuple, List[str]] = {}
    for k, lab in zip(keys, labels):
        groups.setdefault(tuple("nan" if not np.isfinite(x) else x for x in k), []).append(lab)
    return [g for g in groups.values() if len(g) > 1]


def ablation_deltas_figure(analysis, *, title: Optional[str] = None):
    """Mean paired delta (+/- sd over seeds and periods) per term, mode and primary metric, coloured
    by that comparison's verdict. Positive = the term helps."""
    import plotly.graph_objects as go

    rows = []
    for term, t in analysis.get("terms", {}).items():
        for mode, m in t["modes"].items():
            for c in m["metrics"]:
                rows.append((f"{term.replace('LAMBDA_', '')} · {mode.replace('leave_one_', 'one ')} · "
                             f"{c['metric'].split('/')[-1]}", c["mean_delta"], c.get("sd_delta"), c["verdict"]))
    for c in (analysis.get("family") or {}).get("metrics", []):
        rows.append((f"family · {c['metric'].split('/')[-1]}", c["mean_delta"], c.get("sd_delta"), c["verdict"]))
    fig = go.Figure()
    for verdict, color in VERDICT_COLORS.items():
        sel = [r for r in rows if r[3] == verdict]
        if sel:
            fig.add_trace(go.Bar(y=[r[0] for r in sel], x=[r[1] for r in sel], orientation="h", name=verdict,
                                 marker_color=color,
                                 error_x=dict(type="data", array=[r[2] if r[2] == r[2] else 0 for r in sel])))
    fig.add_vline(x=0, line_color="#9a9890")
    fig.update_layout(title=title or "Ablation: paired deltas (positive = the term helps)", barmode="overlay",
                      height=max(360, 22 * len(rows)), xaxis_title="mean delta over (seed, period) pairs")
    return apply(fig)


# ------------------------------------------------------------------ registry entries (data, config)
def runs_comparison(data, config=None, **kw):
    return runs_comparison_figure(data, **kw)


def ablation_deltas(data, config=None, **kw):
    return ablation_deltas_figure(data, **kw)
