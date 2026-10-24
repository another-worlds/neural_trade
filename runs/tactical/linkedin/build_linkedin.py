"""LinkedIn visual set: 7 carousel slides (1080x1350), a 1200x627 cover, a PDF carousel, facts.json and alt_text.md.

usage: python build_linkedin.py [--no-render]
Every number on a slide comes from FACTS below; each is asserted against its source file at build time.
Self-contained HTML (inline SVG/CSS, system fonts only). Rendering: headless Edge, CLI screenshot / print-to-pdf.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # worktree root
MAIN = Path("D:/nt/neural_trade")          # docs/STATUS.md of the MVP branch (not on nt-tactical)
TACT = REPO / "runs" / "tactical"
HYPER = Path("D:/nt/nt_hyperspace_ro")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# The code uses a GRU (src/neural_trade/models/gru_attention.py: Bidirectional(GRU(64))). The post says LSTM.
# Change this one constant to 'Bi-LSTM' to redraw every slide with the post's wording.
RECURRENT_LABEL = "Bi-GRU"

# palette (hyperspace/config.py CSS) -------------------------------------------------------------
BG0, BG1 = "#060b17", "#0f1e3a"
MINT, AMBER, INDIGO, RED, GREEN = "#64ffda", "#fbbf24", "#818cf8", "#f87171", "#34d399"
TXT, MUTED, MUTED2, SKY = "#dce8f0", "#8ab4cc", "#8ea8c2", "#4da6ff"
H_COL = ["#3987e5", "#d95926", "#199e70"]   # visualization/theme.py horizon colours
SANS = "'Segoe UI', Arial, sans-serif"
MONO = "Consolas, 'Courier New', monospace"

# ================================================================================================
# facts: id -> (value, display, source file, quote / key); verified in verify_facts()
# ================================================================================================
LOGP = "runs/tactical/LOG.md"
T09 = "runs/tactical/tasks/T09_regression_anatomy/results_minute.json"
FACTS = {
    "h25_logreg_vs_frozen": (-0.0006, "-0.0006 [-0.006, +0.005]", LOGP, "H25: logreg vs logreg_frozen AUC -0.0006 [-0.006,+0.005]"),
    "h25_logreg_dir": (0.0007, "+0.0007 [-0.006, +0.008]", LOGP, "H25: logreg_dir +0.0007 [-0.006,+0.008]"),
    "h26_thin": (-0.002, "-0.002 [-0.005, +0.001]", LOGP, "H26 A: learned periods do not beat textbook on direction (-0.002 [-0.005,+0.001])"),
    "h26_none": (-0.002, "-0.002 (n.s.)", LOGP, "H26 D: no indicators at all = the base on direction (AUC -0.002 n.s.)"),
    "h23_net": (0.521, "0.521", LOGP, "H23: no-price network 0.521 (0.532 / 0.519 / 0.511) on the same val blocks"),
    "h23_logreg": (0.570, "0.570", LOGP, "H23: check slices 0.570 (0.623 / 0.551 / 0.538)"),
    "nt104_net": (0.535, "0.535", "D:/nt/neural_trade/docs/STATUS.md", "capacity_v1 (NT-104, 5 judgement folds): h1 AUC gru_attention 0.535"),
    "nt104_logreg": (0.552, "0.552", "D:/nt/neural_trade/docs/STATUS.md", "against logreg_lags 0.552 on the same blocks"),
    "h82_hit": (0.503, "50.3%", LOGP, "H82: 2 bps: hit 0.503 (11 months 2025-11..2026-09, 437 trades)"),
    "h82_months": (11, "11 unseen months", LOGP, "H82: scored 2025-11..2026-09 (11 months, 437 trades)"),
    "h82_trades": (437, "437 trades", LOGP, "H82: 437 trades"),
    "h53_2018": (0.286, "0.286", LOGP, "H53: HAR Spearman 0.286 (2018)"),
    "h53_2023": (0.411, "0.411", LOGP, "H53: 0.411 (2023)"),
    "h53_2025": (0.399, "0.399", LOGP, "H53: 0.399 (2025)"),
    "t09_nopen_auc": ("yearly 2017-2025", "0.527 0.562 0.561 0.567 0.553 0.540 0.551 0.533 0.518", T09, "q1.nopen.auc[year].mean"),
    "t09_close_auc": ("yearly 2017-2025", "0.568 0.590 0.599 0.590 0.549 0.543 0.554 0.531 0.521", T09, "q1.close.auc[year].mean"),
    "periods_54": (54, "54 learnable periods", "docs/BACKLOG.md", "NT-047 QA note: 'the default 54 periods' (14 families, D-031/D-047)"),
    "families_14": (14, "14 families", "docs/DECISIONS.md", "D-047: OHLCV input with all 14 indicator families"),
    "heads_8": (8, "8 heads", "src/neural_trade/models/gru_attention.py", "layers.MultiHeadAttention(num_heads=8, key_dim=att_key_dim)(memory, memory)"),
    "horizons": ("10/15/20 min", "10 / 15 / 20 min", "CLAUDE.md", "BTC/USDT 1-minute (60-minute window, 10/15/20-minute horizons)"),
    "window": (60, "60 bars", "CLAUDE.md", "60-minute window"),
    "hyper_dim": (80, "80-dimensional", "D:/nt/nt_hyperspace_ro/hyperspace/config.py", "UKT_FEATURE_DIM: int = 80"),
    "hyper_nodes": (6, "6 nodes", "D:/nt/nt_hyperspace_ro/hyperspace/config.py", "GEOPOLITICAL_NODES (USA, Russia, China, Britain, India, Brazil)"),
    "hyper_edges": (15, "15 edges", "D:/nt/nt_hyperspace_ro/hyperspace/config.py", "GEOPOLITICAL_EDGES (15 tuples)"),
    "hyper_apis": ("32+ keyless APIs", "32+ keyless public APIs", "D:/nt/nt_hyperspace_ro/docs/ARCHITECTURE.md", "Data Sources (32+ keyless APIs)"),
    "hyper_layers": (10, "10 raster layers", "D:/nt/nt_hyperspace_ro/hyperspace/config.py", "REGION_DESCRIPTIONS['geospatial-kernel']: 10 data layers"),
}
_used = set()


def fact(k):
    _used.add(k)
    return FACTS[k][1]


# geometry of the real hyperspace graph (config.py), copied here and verified against the file ----------
NODES = {  # name: (influence, lat, lon, colour)
    "USA": (0.95, 38.9, -77.0, "#3498db"), "Russia": (0.75, 55.8, 37.6, "#e74c3c"),
    "China": (0.90, 39.9, 116.4, "#e67e22"), "Britain": (0.70, 51.5, -0.1, "#2980b9"),
    "India": (0.72, 28.6, 77.2, "#2ecc71"), "Brazil": (0.55, -15.8, -47.9, "#9b59b6")}
EDGES = [("USA", "Britain", 0.92, "alliance"), ("USA", "India", 0.58, "alignment"), ("USA", "China", -0.45, "competition"),
         ("USA", "Russia", -0.72, "competition"), ("USA", "Brazil", 0.35, "alignment"), ("Russia", "China", 0.78, "alignment"),
         ("Russia", "India", 0.40, "alignment"), ("Russia", "Brazil", 0.20, "alignment"), ("Russia", "Britain", -0.65, "competition"),
         ("China", "India", -0.25, "competition"), ("China", "Britain", -0.30, "competition"), ("China", "Brazil", 0.60, "financial_flow"),
         ("Britain", "India", 0.52, "alignment"), ("India", "Brazil", 0.38, "alignment"), ("Brazil", "Britain", 0.25, "alignment")]
ECOL = {"alliance": MINT, "alignment": MINT, "competition": RED, "financial_flow": AMBER}


def verify_facts():
    log = (REPO / LOGP).read_text(encoding="utf-8")
    for key, quote in [("h25_logreg_vs_frozen", "logreg vs logreg_frozen AUC -0.0006 [-0.006,+0.005]"),
                       ("h25_logreg_dir", "logreg_dir +0.0007 [-0.006,+0.008]"),
                       ("h26_thin", "(-0.002 [-0.005,+0.001])"), ("h26_none", "AUC -0.002 n.s.)"),
                       ("h23_net", "no-price network 0.521"), ("h23_logreg", "check slices 0.570"),
                       ("h82_hit", "hit 0.503"), ("h82_months", "(11 months, 437 trades)"),
                       ("h53_2018", "0.286 (2018)"), ("h53_2023", "0.411 (2023)"), ("h53_2025", "0.399 (2025)")]:
        assert quote in log, (key, quote)
    st = (MAIN / "docs/STATUS.md").read_text(encoding="utf-8")
    flat = " ".join(st.split())
    assert "h1 AUC gru_attention 0.535" in flat and "`logreg_lags` 0.552" in flat, "NT-104"
    r = json.loads((REPO / T09).read_text(encoding="utf-8"))
    for k, name in (("nopen", "t09_nopen_auc"), ("close", "t09_close_auc")):
        got = " ".join(f"{r['q1'][k]['auc'][y]['mean']:.3f}" for y in YEARS)
        assert got == FACTS[name][1], (k, got)
    d = json.loads((REPO / "runs/tactical/lab/decay_vol.json").read_text(encoding="utf-8"))
    by = {}
    for row in d:
        by.setdefault(row["month"][:4], []).append(row["har"])
    for y, want in (("2018", 0.286), ("2023", 0.411), ("2025", 0.399)):
        assert round(sum(by[y]) / len(by[y]), 3) == want, y
    cfg = (HYPER / "hyperspace/config.py").read_text(encoding="utf-8")
    assert "UKT_FEATURE_DIM: int = 80" in cfg
    for n, (inf, lat, lon, _) in NODES.items():
        assert f'"{n}"' in cfg and f"influence={inf}" in cfg and f"lat={lat}" in cfg and f"lon={lon}" in cfg, n
    import re
    real = [(m[0], m[1], float(m[2]), m[3]) for m in re.findall(r'\("(\w+)",\s*"(\w+)",\s*(-?[\d.]+),\s*"(\w+)"', cfg)]
    assert real == EDGES, "edges differ from hyperspace/config.py"
    arch = (HYPER / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "32+ keyless APIs" in arch
    gru = (REPO / "src/neural_trade/models/gru_attention.py").read_text(encoding="utf-8")
    assert "Bidirectional(layers.GRU(64" in gru and "MultiHeadAttention(num_heads=8" in gru
    return by


YEARS = [str(y) for y in range(2017, 2026)]

# ================================================================================================
# shared CSS / frame
# ================================================================================================
CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:{BG0}}}
body{{font-family:{SANS};color:{TXT};-webkit-font-smoothing:antialiased}}
.slide{{position:relative;width:1080px;height:1350px;overflow:hidden;padding:56px 60px 0 60px;
  background:radial-gradient(900px 700px at 85% 8%, rgba(100,255,218,.07), transparent 60%),
             radial-gradient(900px 800px at 0% 100%, rgba(129,140,248,.10), transparent 60%),
             linear-gradient(165deg,{BG0} 0%,#0a1428 55%,{BG1} 100%);page-break-after:always;break-after:page}}
.slide:last-child{{page-break-after:auto;break-after:auto}}
.top{{display:flex;justify-content:space-between;align-items:center;height:44px}}
.chip{{font-size:22px;font-weight:700;letter-spacing:.5px;padding:6px 18px;border-radius:999px;border:1px solid}}
.chip.nt{{color:{AMBER};background:#1c0e00;border-color:#6b3800}}
.chip.hs{{color:{MINT};background:#041d12;border-color:#0a3d28}}
.chip.pr{{color:{AMBER};background:#1c0e00;border-color:#6b3800}}
.num{{font-family:{MONO};font-size:24px;color:{MUTED2}}}
h1{{font-size:64px;line-height:1.08;font-weight:800;letter-spacing:-.5px;margin-top:30px;color:{TXT}}}
h1 em{{font-style:normal;color:{MINT}}}
h1 em.a{{color:{AMBER}}}
.sub{{font-size:30px;line-height:1.35;color:{MUTED};margin-top:18px}}
.sub b{{color:{TXT};font-weight:600}}
.vis{{margin-top:26px}}
.foot{{position:absolute;left:60px;right:60px;bottom:34px;display:flex;justify-content:space-between;
  font-family:{MONO};font-size:22px;color:{MUTED2};border-top:1px solid #182135;padding-top:14px}}
svg text{{font-family:{SANS}}} svg .m{{font-family:{MONO}}}
"""


def head(kind, n, h1, sub):
    cls, name = ("nt", "neural_trade") if kind == "nt" else ("hs", "hyperspace")
    return (f'<div class="top"><span class="chip {cls}">{name}</span><span class="num">{n} / 7</span></div>'
            f'<h1>{h1}</h1><div class="sub">{sub}</div>')


def foot(left, right):
    return f'<div class="foot"><span>{left}</span><span>{right}</span></div>'


def slide(inner):
    return f'<div class="slide">{inner}</div>'


def page(body, w=1080, h=1350, extra_css=""):
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>LinkedIn slide</title><style>{CSS}{extra_css}'
            f'@page{{size:{w}px {h}px;margin:0}}</style></head><body>{body}</body></html>')


DEFS = f"""<defs>
<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter>
<filter id="glow2" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3"/></filter>
<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>
<marker id="aha" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{AMBER}"/></marker>
</defs>"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tx(x, y, s, size=24, fill=TXT, anchor="start", weight=400, mono=False, extra=""):
    cls = ' class="m"' if mono else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"'
            f'{cls} {extra}>{esc(s)}</text>')


def card(x, y, w, h, color, side="top", fill="#0b1525", rx=10):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="#1a2e50"/>'
    if side == "top":
        s += f'<path d="M{x+rx},{y+1} H{x+w-rx}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
    else:
        s += f'<path d="M{x+1},{y+rx} V{y+h-rx}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>'
    return s


# ================================================================================================
# the 6-node network (real positions and edges)
# ================================================================================================
LABEL_OFF = {"USA": (0, 1, "middle"), "Russia": (0, -1, "middle"), "China": (0, -1, "middle"),
             "Britain": (-1, -1, "end"), "India": (0, 1, "middle"), "Brazil": (0, 1, "middle")}


def node_xy(name, W, H, pad):
    _, lat, lon, _ = NODES[name]
    x = pad + (lon + 85) / 210 * (W - 2 * pad)
    y = pad + (62 - lat) / 82 * (H - 2 * pad)
    return x, y


def network(W, H, pad=70, rscale=26, label=True, lsize=24, dash=True, gratic=True, opacity=1.0):
    o = [f'<g opacity="{opacity}">']
    if gratic:
        for lon in range(-60, 121, 30):
            x = pad + (lon + 85) / 210 * (W - 2 * pad)
            o.append(f'<line x1="{x:.1f}" y1="{pad-20}" x2="{x:.1f}" y2="{H-pad+20}" stroke="#1a2e50" stroke-width="1" opacity=".55"/>')
        for lat in range(-20, 61, 20):
            y = pad + (62 - lat) / 82 * (H - 2 * pad)
            o.append(f'<line x1="{pad-30}" y1="{y:.1f}" x2="{W-pad+30}" y2="{y:.1f}" stroke="#1a2e50" stroke-width="1" opacity=".55"/>')
    for a, b, w, t in EDGES:
        (x1, y1), (x2, y2) = node_xy(a, W, H, pad), node_xy(b, W, H, pad)
        c = ECOL[t]
        sw = 1.5 + 5 * abs(w)
        o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{sw+5:.1f}" opacity=".16" filter="url(#glow2)"/>')
        da = ' stroke-dasharray="14 9"' if t == "competition" else ""
        o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{sw:.1f}" opacity=".75"{da}/>')
        if dash:  # static "signal pulses"
            o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-dasharray="1 26" opacity=".85"/>')
    for n, (inf, lat, lon, col) in NODES.items():
        x, y = node_xy(n, W, H, pad)
        r = rscale * inf
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r*1.9:.1f}" fill="{col}" opacity=".35" filter="url(#glow)"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{col}" stroke="{TXT}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r*0.38:.1f}" fill="#ffffff" opacity=".55"/>')
        if label:
            dx, dy, anc = LABEL_OFF[n]
            lx = x + dx * (r + 8)
            ly = y + (r + lsize + 4) if dy > 0 else y - r - 10
            o.append(tx(round(lx, 1), round(ly, 1), n, lsize, TXT, anc, 600))
    o.append("</g>")
    return "".join(o)


# ================================================================================================
# slide 1: hero (also used for the 1200x627 cover)
# ================================================================================================
def hero_svg(W, H, big=True):
    rnd = random.Random(7)
    L = int(W * 0.40)
    ymid = H * 0.5
    pts, y = [], 0.0
    n = 90
    for i in range(n):
        y += rnd.gauss(0, 1) + 0.04
        pts.append(y)
    lo, hi = min(pts), max(pts)
    sc = (H * 0.34) / (hi - lo)
    xs = [30 + i * (L - 30) / (n - 1) for i in range(n)]
    ys = [ymid + H * 0.12 - (p - lo) * sc for p in pts]
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    o = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{DEFS}',
         f'<defs><linearGradient id="fade" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{AMBER}" stop-opacity="0"/>'
         f'<stop offset=".35" stop-color="{AMBER}" stop-opacity=".25"/><stop offset="1" stop-color="{AMBER}" stop-opacity="1"/></linearGradient></defs>']
    for i in range(380):  # noise cloud, denser on the left
        t = rnd.random() ** 1.7
        x = 30 + (1 - t) * (L - 30) * 0.98
        yy = ymid + rnd.gauss(0, H * 0.17) * (0.5 + 1.1 * (1 - x / L))
        o.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="{rnd.choice([1.4, 1.8, 2.4])}" fill="{INDIGO}" opacity="{0.12 + 0.45 * (1 - x / L):.2f}"/>')
    o.append(f'<path d="{path}" fill="none" stroke="{AMBER}" stroke-width="9" opacity=".22" filter="url(#glow2)" stroke-linejoin="round"/>')
    o.append(f'<path d="{path}" fill="none" stroke="url(#fade)" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
    o.append(tx(30, H - 14, "price: signal dissolving into noise (illustration)", 20 if big else 16, MUTED2))
    # link from the line's end to the network
    nx0 = int(W * 0.44)
    x1, y1 = xs[-1], ys[-1]
    o.append(f'<path d="M{x1:.1f},{y1:.1f} C{x1+90:.1f},{y1:.1f} {nx0:.1f},{ymid:.1f} {nx0+50:.1f},{ymid-20:.1f}" fill="none" stroke="{MINT}" stroke-width="2" stroke-dasharray="3 9" stroke-linecap="round" opacity=".8"/>')
    o.append(f'<g transform="translate({nx0},0)">{network(W - nx0, H, pad=int(H*0.12), rscale=int(H*0.058), lsize=22 if big else 17, label=True)}</g>')
    o.append("</svg>")
    return "".join(o)


def slide1():
    inner = (head("nt", 1, "Do <em>whales</em> leave signatures in technical indicators?",
                  "I built a network to find out. Then I changed the question.")
             + f'<div class="vis" style="margin-top:34px">{hero_svg(960, 820)}</div>'
             + foot("neural_trade &rarr; hyperspace", "illustration, no data"))
    return slide(inner)


def cover():
    inner = (f'<div style="position:absolute;left:44px;top:40px;width:560px">'
             f'<span class="chip nt">neural_trade</span> <span class="num" style="margin:0 6px">&rarr;</span> <span class="chip hs">hyperspace</span>'
             f'<div style="font-size:50px;line-height:1.1;font-weight:800;margin-top:30px;letter-spacing:-.4px">Do <span style="color:{MINT}">whales</span> leave signatures in technical indicators?</div>'
             f'<div style="font-size:26px;color:{MUTED};margin-top:22px;line-height:1.35">What a learned-indicator network taught me, and why I moved to simulations.</div></div>'
             f'<div style="position:absolute;left:560px;top:40px">{hero_svg(660, 540, big=False)}</div>')
    css = ".slide{width:1200px;height:627px;padding:0}"
    return page(f'<div class="slide">{inner}</div>', 1200, 627, css)


# ================================================================================================
# slide 2: the model
# ================================================================================================
def slide2():
    W = 960
    bw = 880
    o = [f'<svg width="{W}" height="900" viewBox="0 0 {W} 900" xmlns="http://www.w3.org/2000/svg">{DEFS}']
    # input
    o.append(card(0, 0, bw, 66, SKY, "left"))
    o.append(tx(24, 43, "OHLCV window", 29, TXT, weight=700))
    o.append(tx(bw - 20, 43, f"{FACTS['window'][1]} of 1-minute bars", 24, MUTED, "end", mono=True))
    o.append(f'<path d="M{bw/2},68 V104" stroke="{MUTED}" stroke-width="3" marker-end="url(#ah)"/>')
    # learnable indicators
    y0 = 108
    o.append(card(0, y0, bw, 226, INDIGO))
    o.append(tx(24, y0 + 46, "Learnable indicators", 33, INDIGO, weight=700))
    o.append(tx(24, y0 + 86, f"{fact('families_14')} x 3 instances, {fact('periods_54')}", 26, TXT))
    o.append(tx(24, y0 + 128, "alpha = sigmoid(logit)", 25, AMBER, mono=True))
    o.append(tx(24, y0 + 162, "period = 2/alpha - 1", 25, AMBER, mono=True))
    o.append(tx(24, y0 + 198, "an exponential smoother is a 1-D convolution", 22, MUTED))
    # kernel sketch: two exponential kernels of different width
    kx, ky = 560, y0 + 190
    for wid, col, dashed in ((34, MUTED2, True), (92, AMBER, False)):
        pts = " ".join(f"{kx + t:.1f},{ky - 110 * math.exp(-t / wid):.1f}" for t in range(0, 300, 6))
        dd = ' stroke-dasharray="7 6"' if dashed else ""
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3.5"{dd}/>')
    o.append(f'<line x1="{kx}" y1="{ky}" x2="{kx+300}" y2="{ky}" stroke="#2a4066" stroke-width="2"/>')
    o.append(tx(kx + 300, y0 + 40, "kernel width is learned", 22, MUTED, "end"))
    o.append(f'<path d="M{bw/2},{y0+228} V{y0+262}" stroke="{MUTED}" stroke-width="3" marker-end="url(#ah)"/>')
    # recurrent + attention
    y1 = 376
    o.append(card(0, y1, bw, 116, SKY))
    o.append(f'<rect x="24" y="{y1+28}" width="360" height="62" rx="10" fill="#0d1b2a" stroke="#1a3a5c"/>')
    o.append(tx(204, y1 + 72, RECURRENT_LABEL, 36, TXT, "middle", 700))
    o.append(f'<path d="M388,{y1+59} H430" stroke="{MUTED}" stroke-width="3" marker-end="url(#ah)"/>')
    o.append(f'<rect x="436" y="{y1+28}" width="420" height="62" rx="10" fill="#0d1b2a" stroke="#1a3a5c"/>')
    o.append(tx(646, y1 + 72, f"attention, {fact('heads_8')}", 32, TXT, "middle", 700))
    o.append(f'<path d="M{bw/2},{y1+118} V{y1+150}" stroke="{MUTED}" stroke-width="3" marker-end="url(#ah)"/>')
    # heads
    y2 = 536
    cw, gap = 284, 14
    for i, (h, col) in enumerate(zip(("10 min", "15 min", "20 min"), H_COL)):
        x = i * (cw + gap)
        o.append(card(x, y2, cw, 168, col))
        o.append(tx(x + 20, y2 + 44, f"horizon {h}", 29, col, weight=700))
        for j, lab in enumerate(("price change", "P(up)", "variance")):
            yy = y2 + 82 + j * 32
            o.append(f'<circle cx="{x+28}" cy="{yy-8}" r="6" fill="{col}"/>')
            o.append(tx(x + 46, yy, lab, 26, TXT))
    o.append(tx(0, y2 + 200, "variance = the model's confidence in each horizon", 22, MUTED))
    # losses
    ly = 764
    L = [("direction BCE", "trains P(up)", GREEN, 0, 270), ("Gaussian NLL + CRPS", "train confidence", AMBER, 286, 300),
         ("physics-inspired", "T-perp, Casimir, HD,|IFE, vacuum", INDIGO, 602, 278)]
    for t, s, col, x, w in L:
        o.append(card(x, ly, w, 112, col, "top", "#0d1232" if col == INDIGO else "#0b1525"))
        o.append(tx(x + 16, ly + 42, t, 26 if len(t) < 21 else 24, col, weight=700))
        for k_, part in enumerate(s.split("|")):
            o.append(tx(x + 16, ly + 76 + k_ * 26, part, 22, MUTED))
    # gradient arrow (backwards)
    o.append(f'<path d="M{bw+4},{ly+80} H{bw+34} Q{bw+42},{ly+80} {bw+42},{ly+70} V{y0+60} Q{bw+42},{y0+50} {bw+30},{y0+50} H{bw+4}" fill="none" stroke="{AMBER}" stroke-width="4" stroke-dasharray="12 8" marker-end="url(#aha)" stroke-linecap="round"/>')
    o.append(f'<g transform="translate({bw+70},{(ly+y0)/2+40}) rotate(-90)">{tx(0, 0, "gradients flow back to the periods", 24, AMBER, "middle", 700)}</g>')
    o.append("</svg>")
    inner = (head("nt", 2, "A network that learns its own <em class=\"a\">indicator settings</em>",
                  "Gradients from the losses reach the periods themselves.")
             + f'<div class="vis" style="margin-top:20px">{"".join(o)}</div>'
             + foot("src/neural_trade/models/gru_attention.py", "reference setup: BTC/USDT 1-min"))
    return slide(inner)


# ================================================================================================
# slide 3: forest plot
# ================================================================================================
def slide3():
    W, X0, X1, DOM = 960, 70, 890, 0.012
    sx = lambda v: (X0 + X1) / 2 + v / DOM * (X1 - X0) / 2
    rows = [("Learned periods vs textbook", "logistic read-out in the network (H25)", -0.0006, -0.006, 0.005, "h25_logreg_vs_frozen", INDIGO),
            ("Periods trained by direction only", "vs textbook, same read-out (H25)", 0.0007, -0.006, 0.008, "h25_logreg_dir", INDIGO),
            ("Learned vs frozen, thin read-out", "linear_indicators (H26)", -0.002, -0.005, 0.001, "h26_thin", AMBER),
            ("No indicators at all vs 14 families", "difference stated, no interval in the journal (H26)", -0.002, None, None, "h26_none", AMBER)]
    rh, y0 = 156, 90
    H = y0 + len(rows) * rh + 130
    o = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{DEFS}']
    zx = sx(0)
    o.append(f'<rect x="{sx(-0.0015):.1f}" y="{y0-30}" width="{sx(0.0015)-sx(-0.0015):.1f}" height="{len(rows)*rh+30}" fill="{MINT}" opacity=".07"/>')
    o.append(f'<line x1="{zx}" y1="{y0-30}" x2="{zx}" y2="{y0+len(rows)*rh}" stroke="{MINT}" stroke-width="3" stroke-dasharray="8 6"/>')
    o.append(tx(zx, 28, "no difference", 28, MINT, "middle", 700))
    o.append(tx(zx, 58, "0", 24, MINT, "middle", mono=True))
    for i, (lab, sub, v, lo, hi, key, col) in enumerate(rows):
        y = y0 + i * rh
        o.append(tx(X0 - 10, y + 22, lab, 27, TXT, weight=700))
        o.append(tx(X0 - 10, y + 50, sub, 21, MUTED))
        yy = y + 112
        o.append(f'<line x1="{X0}" y1="{yy}" x2="{X1}" y2="{yy}" stroke="#16263f" stroke-width="2"/>')
        if lo is not None:
            o.append(f'<line x1="{sx(lo):.1f}" y1="{yy}" x2="{sx(hi):.1f}" y2="{yy}" stroke="{col}" stroke-width="7" stroke-linecap="round"/>')
            for e in (lo, hi):
                o.append(f'<line x1="{sx(e):.1f}" y1="{yy-12}" x2="{sx(e):.1f}" y2="{yy+12}" stroke="{col}" stroke-width="4"/>')
            o.append(f'<circle cx="{sx(v):.1f}" cy="{yy}" r="12" fill="{col}" stroke="{TXT}" stroke-width="3"/>')
        else:
            o.append(f'<circle cx="{sx(v):.1f}" cy="{yy}" r="12" fill="{BG0}" stroke="{col}" stroke-width="4"/>')
        o.append(tx(X1, y + 22, fact(key), 24, col, "end", mono=True))
    ay = y0 + len(rows) * rh + 14
    o.append(f'<line x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}" stroke="#2a4066" stroke-width="2"/>')
    for t in (-0.01, -0.005, 0, 0.005, 0.01):
        o.append(f'<line x1="{sx(t):.1f}" y1="{ay}" x2="{sx(t):.1f}" y2="{ay+10}" stroke="#2a4066" stroke-width="2"/>')
        o.append(tx(sx(t), ay + 38, f"{t:+.3f}" if t else "0", 21, MUTED, "middle", mono=True))
    o.append(tx(X0, ay + 76, "<- learned worse", 22, MUTED))
    o.append(tx(X1, ay + 76, "learned better ->", 22, MUTED, "end"))
    o.append(tx(zx, ay + 76, "difference in direction AUC (95% interval)", 22, TXT, "middle"))
    o.append("</svg>")
    inner = (head("nt", 3, "Learned periods =<br><em class=\"a\">textbook</em> periods",
                  "Did the network find better indicator settings than the textbook? Every interval shown crosses zero.")
             + f'<div class="vis">{"".join(o)}</div>'
             + foot("tactical journal rows H25, H26", "BTC 1-min, long blocks"))
    return slide(inner)


# ================================================================================================
# slide 4: yearly AUC
# ================================================================================================
def slide4():
    r = json.loads((REPO / T09).read_text(encoding="utf-8"))["q1"]
    no = [r["nopen"]["auc"][y] for y in YEARS]
    cl = [r["close"]["auc"][y]["mean"] for y in YEARS]
    W, H = 960, 560
    L, R, T, B = 80, 40, 20, 70
    ymin, ymax = 0.48, 0.62
    px = lambda i: L + i * (W - L - R) / 8
    py = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - B)
    o = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{DEFS}']
    for v in (0.50, 0.52, 0.54, 0.56, 0.58, 0.60):
        o.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{W-R}" y2="{py(v):.1f}" stroke="#16263f" stroke-width="1.5"/>')
        o.append(tx(L - 12, py(v) + 8, f"{v:.2f}", 21, MUTED, "end", mono=True))
    band = " ".join(f"{px(i):.1f},{py(d['hi']):.1f}" for i, d in enumerate(no)) + " " + " ".join(f"{px(i):.1f},{py(d['lo']):.1f}" for i, d in reversed(list(enumerate(no))))
    o.append(f'<polygon points="{band}" fill="{AMBER}" opacity=".16"/>')
    o.append(f'<line x1="{L}" y1="{py(0.5):.1f}" x2="{W-R}" y2="{py(0.5):.1f}" stroke="{RED}" stroke-width="3.5" stroke-dasharray="10 7"/>')
    o.append(tx(L + 6, py(0.5) + 30, "coin flip = 0.50", 24, RED, "start", 700))
    cpts = " ".join("%.1f,%.1f" % (px(i), py(v)) for i, v in enumerate(cl))
    npts = " ".join("%.1f,%.1f" % (px(i), py(d["mean"])) for i, d in enumerate(no))
    o.append(f'<polyline points="{cpts}" fill="none" stroke="{INDIGO}" stroke-width="3" stroke-dasharray="3 7" stroke-linecap="round"/>')
    o.append(f'<polyline points="{npts}" fill="none" stroke="{AMBER}" stroke-width="6" stroke-linejoin="round"/>')
    for i, d in enumerate(no):
        o.append(f'<circle cx="{px(i):.1f}" cy="{py(d["mean"]):.1f}" r="8" fill="{AMBER}" stroke="{BG0}" stroke-width="3"/>')
        ly_ = py(d["hi"]) - 14 if i == 8 else py(d["lo"]) + 32
        o.append(tx(px(i), ly_, f"{d['mean']:.3f}"[1:], 21, AMBER, "middle", mono=True))
        o.append(tx(px(i), H - B + 34, YEARS[i], 23, MUTED, "middle", mono=True))
    o.append(f'<line x1="{L+10}" y1="{T+12}" x2="{L+60}" y2="{T+12}" stroke="{AMBER}" stroke-width="6"/>')
    o.append(tx(L + 72, T + 20, "next-open fill, 95% interval over months", 22, TXT))
    o.append(f'<line x1="{L+10}" y1="{T+44}" x2="{L+60}" y2="{T+44}" stroke="{INDIGO}" stroke-width="3" stroke-dasharray="3 7"/>')
    o.append(tx(L + 72, T + 52, "close fill (optimistic)", 22, MUTED))
    o.append(tx(W / 2, H - 6, "direction AUC of a 13-feature logistic regression, monthly walk-forward", 21, MUTED, "middle"))
    o.append("</svg>")
    cards = [("0.521 vs 0.570", "network vs logistic regression on the same long blocks (H23)", RED),
             ("0.535 vs 0.552", "network vs logistic on lags, h1 AUC, 5 judgement folds <span style=\"white-space:nowrap\">(NT-104)</span>", RED),
             ("50.3% hit rate", "forward test, 11 unseen months, 437 trades (H82)", AMBER)]
    cs = "".join(f'<div style="flex:1;background:#0b1525;border:1px solid #1a2e50;border-top:4px solid {c};border-radius:10px;padding:14px 16px">'
                 f'<div style="font-family:{MONO};font-size:34px;font-weight:700;color:{TXT}">{a}</div>'
                 f'<div style="font-size:21px;color:{MUTED};margin-top:6px;line-height:1.3">{b}</div></div>' for a, b, c in cards)
    for k in ("h23_net", "h23_logreg", "nt104_net", "nt104_logreg", "h82_hit", "h82_months", "h82_trades", "t09_nopen_auc", "t09_close_auc"):
        _used.add(k)
    inner = (head("nt", 4, "Even the best simple model drifted to a <em class=\"a\">coin flip</em>",
                  "The network scored lower still.")
             + f'<div class="vis" style="margin-top:14px">{"".join(o)}</div>'
             + f'<div style="display:flex;gap:14px;margin-top:10px">{cs}</div>'
             + foot("T09 regression anatomy, H23, H82, NT-104", "BTC/USD 1-min bars, 2017-2025"))
    return slide(inner)


# ================================================================================================
# slide 5: lesson
# ================================================================================================
def slide5(by):
    W, H = 960, 640
    layers = [("news", SKY), ("GIS / terrain", GREEN), ("elections", INDIGO), ("shareholder votes", AMBER),
              ("finance and trade", "#e67e22"), ("social graph", "#c084fc"), ("physical events", RED)]
    o = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{DEFS}']
    n = len(layers)
    step = 50
    for i, (lab, col) in enumerate(layers):
        y = 20 + i * step
        o.append(f'<polygon points="150,{y+56} 380,{y} 800,{y} 570,{y+56}" fill="{col}" opacity=".30" stroke="{col}" stroke-width="2.5"/>')
        o.append(tx(475, y + 36, lab, 25, TXT, "middle", 700))
    ytop = 20 + (n - 1) * step + 56
    # rays converging to the lens
    lx, ly = 480, 530
    for xs in (190, 330, 480, 640, 760):
        o.append(f'<line x1="{xs}" y1="{ytop-6}" x2="{lx}" y2="{ly-18}" stroke="{TXT}" stroke-width="1.6" opacity=".4"/>')
    o.append(f'<ellipse cx="{lx}" cy="{ly}" rx="190" ry="24" fill="{MINT}" opacity=".18" stroke="{MINT}" stroke-width="3"/>')
    o.append(f'<ellipse cx="{lx}" cy="{ly}" rx="190" ry="24" fill="none" stroke="{MINT}" stroke-width="10" opacity=".18" filter="url(#glow2)"/>')
    o.append(f'<polygon points="{lx-170},{ly+10} {lx+170},{ly+10} {lx+6},{ly+84} {lx-6},{ly+84}" fill="{MINT}" opacity=".16"/>')
    o.append(f'<line x1="{lx}" y1="{ly+24}" x2="{lx}" y2="{ly+84}" stroke="{MINT}" stroke-width="2.5"/>')
    o.append(tx(lx + 205, ly + 6, "projection", 24, MINT, weight=700))
    # the price line: thin, 1-D
    o.append(f'<line x1="60" y1="{ly+84}" x2="900" y2="{ly+84}" stroke="#2a4066" stroke-width="2" stroke-dasharray="3 8"/>')
    rnd = random.Random(3)
    yv, pts = 0, []
    for i in range(70):
        yv += rnd.gauss(0, 3.2)
        pts.append(f"{60 + i * 12:.0f},{ly + 84 + yv:.1f}")
    o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{AMBER}" stroke-width="4" stroke-linejoin="round"/>')
    o.append(tx(60, ly + 128, "price: one number per minute", 24, AMBER, weight=700))
    o.append(tx(900, ly + 128, "inputs a price-only model never sees  (layers above)", 21, MUTED, "end"))
    o.append("</svg>")
    # callout with sparkline
    har = [sum(by[y]) / len(by[y]) for y in YEARS]
    sw, sh = 300, 90
    sp = " ".join(f"{i*sw/8:.1f},{sh - (v-0.25)/(0.45-0.25)*sh:.1f}" for i, v in enumerate(har))
    spark = f'<svg width="{sw+20}" height="{sh+16}" viewBox="-10 -8 {sw+20} {sh+16}"><polyline points="{sp}" fill="none" stroke="{MINT}" stroke-width="5" stroke-linejoin="round"/><circle cx="{sw}" cy="{sh-(har[-1]-0.25)/0.2*sh:.1f}" r="7" fill="{MINT}"/></svg>'
    for k in ("h53_2018", "h53_2023", "h53_2025"):
        _used.add(k)
    call = (f'<div style="display:flex;gap:22px;align-items:center;margin-top:12px;background:#06111f;border:1px solid #1a2e50;border-left:5px solid {MINT};border-radius:0 10px 10px 0;padding:16px 20px">'
            f'<div style="flex:1"><div style="font-size:27px;line-height:1.3;color:{TXT}">The <b style="color:{MINT}">size</b> of the next move kept structure while its <b style="color:{AMBER}">sign</b> faded.</div>'
            f'<div style="font-size:22px;color:{MUTED};margin-top:8px;line-height:1.35">Volatility predictability (Spearman, HAR model): <span style="font-family:{MONO};color:{TXT}">0.286</span> in 2018, <span style="font-family:{MONO};color:{TXT}">0.399</span> in 2025 (peak 0.411 in 2023).</div></div>'
            f'<div>{spark}<div style="font-family:{MONO};font-size:19px;color:{MUTED2};text-align:center">2017 - 2025</div></div></div>')
    inner = (head("nt", 5, "A price is a 1-D shadow of a <em>multimodal</em> world",
                  "So predicting it from past prices alone makes little sense.")
             + f'<div class="vis" style="margin-top:10px">{"".join(o)}</div>' + call
             + foot("tactical journal H53, lab/decay_vol.json", "layers: conceptual"))
    return slide(inner)


# ================================================================================================
# slide 6: hyperspace
# ================================================================================================
BLOCKS = [("temporal-pattern", "TFT forecasts of country ETFs", SKY),
          ("semantic-embedding", "BERTopic over news", INDIGO),
          ("structural-centrality", "geopolitical graph", MINT),
          ("dynamic-agent", "agent simulation: resources, alliance eigenvalues", AMBER),
          ("geospatial-kernel", "10 raster layers at the 6 nodes", GREEN)]


def slide6():
    W = 960
    NH = 380
    o = [f'<svg width="{W}" height="1000" viewBox="0 0 {W} 1000" xmlns="http://www.w3.org/2000/svg">{DEFS}']
    # network panel
    o.append(card(0, 0, W, NH, MINT))
    o.append(f'<g transform="translate(0,14)">{network(W, NH - 24, pad=66, rscale=32, lsize=25)}</g>')
    o.append(tx(22, 40, "6 nodes, 15 edges, node size = influence", 22, MUTED))
    lg = [("alliance / alignment", MINT, ""), ("competition", RED, "10 6"), ("financial flow", AMBER, "")]
    for i, (t, c, d) in enumerate(lg):
        yy = NH - 88 + i * 28
        da = ' stroke-dasharray="%s"' % d if d else ""
        o.append(f'<line x1="{W-250}" y1="{yy-6}" x2="{W-208}" y2="{yy-6}" stroke="{c}" stroke-width="5"{da}/>')
        o.append(tx(W - 198, yy, t, 21, TXT))
    # blocks: 3 + 2 grid
    bw, bh, gap = 304, 100, 24
    pos = [(i * (bw + gap), 400) for i in range(3)] + [(bw // 2 + gap // 2 + i * (bw + gap), 400 + bh + 16) for i in range(2)]
    for (n, d, c), (x, y) in zip(BLOCKS, pos):
        o.append(card(x, y, bw, bh, c))
        o.append(tx(x + 16, y + 38, n, 24, c, weight=700))
        words, line, lines = d.split(), "", []
        for w_ in words:
            if len(line) + len(w_) > 29:
                lines.append(line); line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for j, ln in enumerate(lines[:2]):
            o.append(tx(x + 16, y + 68 + j * 26, ln, 21, MUTED))
    # UKT
    uy = 400 + 2 * bh + 16 + 40
    o.append(f'<path d="M{W/2},{uy-36} V{uy-4}" stroke="{MUTED}" stroke-width="3" marker-end="url(#ah)"/>')
    o.append(f'<rect x="0" y="{uy}" width="{W}" height="118" rx="12" fill="#0b1525" stroke="#1a2e50"/>')
    sw_ = (W - 24) / 5
    for i, (n, d, c) in enumerate(BLOCKS):
        o.append(f'<rect x="{12+i*sw_:.1f}" y="{uy+66}" width="{sw_-4:.1f}" height="14" rx="4" fill="{c}" opacity=".85"/>')
    o.append(tx(W / 2, uy + 42, f"one {fact('hyper_dim')} Universal Knowledge Tensor", 29, TXT, "middle", 700))
    o.append(tx(W - 14, uy + 106, "stripes = block provenance labels, not to scale", 19, MUTED2, "end"))
    # stages
    sy = uy + 156
    o.append(f'<path d="M125,{uy+120} V{sy-4}" stroke="{MUTED}" stroke-width="3" marker-end="url(#ah)"/>')
    st = [("SVD kernels", "", INDIGO, 0, 250), ("sparse-autoencoder", "concepts", AMBER, 274, 300), ("semantic canvas +", "governance flags", MINT, 598, 362)]
    for a_, b_, c, x, w in st:
        o.append(card(x, sy, w, 98, c))
        if b_:
            o.append(tx(x + w / 2, sy + 44, a_, 25, TXT, "middle", 700))
            o.append(tx(x + w / 2, sy + 76, b_, 25, TXT, "middle", 700))
        else:
            o.append(tx(x + w / 2, sy + 60, a_, 25, TXT, "middle", 700))
        if x < 598:
            o.append(f'<path d="M{x+w+2},{sy+49} H{x+w+20}" stroke="{MUTED}" stroke-width="3" marker-end="url(#ah)"/>')
    o.append(tx(0, sy + 138, f"Data: {fact('hyper_apis')} (docs/ARCHITECTURE.md).", 22, MUTED))
    o.append(tx(0, sy + 168, "Political block: political-science datasets and a geopolitical graph.", 22, MUTED))
    o.append("</svg>")
    inner = (f'<div class="top"><span class="chip hs">hyperspace</span><span class="chip pr">prototype</span><span class="num">6 / 7</span></div>'
             f'<h1 style="margin-top:20px;font-size:58px">Five blocks of the world, one <em>shared</em> tensor</h1>'
             f'<div class="vis" style="margin-top:16px">{"".join(o)}</div>'
             + foot("hyperspace/config.py, ARCHITECTURE.md", "node layout: lat/lon"))
    return slide(inner)


# ================================================================================================
# slide 7: close
# ================================================================================================
def slide7():
    items = [("news", SKY), ("GIS layers", GREEN), ("election results across democracies", INDIGO), ("shareholder votes across companies", AMBER)]
    rows = "".join(f'<div style="display:flex;align-items:center;justify-content:space-between;gap:16px;padding:20px 24px;background:#0b1525;border:1px solid #1a2e50;border-left:6px solid {c};border-radius:0 10px 10px 0;margin-top:14px">'
                   f'<span style="font-size:32px;font-weight:600;line-height:1.2">{t}</span>'
                   f'<span style="font-family:{MONO};font-size:21px;color:{AMBER};background:#1c0e00;border:1px solid #6b3800;border-radius:999px;padding:4px 14px;white-space:nowrap">planned</span></div>'
                   for t, c in items)
    bg = f'<svg style="position:absolute;left:60px;top:955px;opacity:.6" width="960" height="300" viewBox="0 0 960 300">{DEFS}{network(960, 300, pad=50, rscale=24, label=True, lsize=20, gratic=True)}</svg>'
    inner = (head("hs", 7, "I do not know if it will work.<br><em>I am optimistic.</em>", "")
             + bg
             + f'<div style="position:relative;margin-top:34px"><div style="font-size:26px;color:{MUTED};letter-spacing:.5px;text-transform:uppercase;font-weight:700">Next inputs</div>{rows}</div>'
             + f'<div style="position:relative;margin-top:44px;background:linear-gradient(135deg,#050c1a,#09152a);border:1px solid #1a2e50;border-top:4px solid {MINT};border-radius:10px;padding:26px 28px">'
             f'<div style="font-size:24px;color:{MUTED}">Simulation prototype</div>'
             f'<div style="font-family:{MONO};font-size:34px;color:{MINT};margin-top:8px;font-weight:700">github.com/another-worlds/hyperspace</div></div>'
             + foot("neural_trade &rarr; hyperspace", "7 / 7"))
    return slide(inner)


# ================================================================================================
# alt text
# ================================================================================================
ALT = [
    "Dark navy hero image. On the left a price line dissolves into a cloud of noise. A dotted line leads to the right, where six glowing nodes (USA, Russia, China, Britain, India, Brazil) are linked as a network. Headline: Do whales leave signatures in technical indicators?",
    f"Schematic of the model. An OHLCV window feeds learnable indicators (14 families, 54 learnable periods, each a trainable logit; an exponential smoother is a 1-D convolution with a learned width). Then a {RECURRENT_LABEL} and 8-head attention, then heads for 10, 15 and 20 minutes (price change, P(up), variance). Losses: direction BCE, Gaussian NLL plus CRPS, physics-inspired terms. A dashed amber arrow shows gradients flowing back to the periods.",
    "Forest plot of differences in direction AUC with zero marked as no difference. Learned minus textbook periods: -0.0006 [-0.006, +0.005]; periods trained by direction only: +0.0007 [-0.006, +0.008]; thin read-out learned versus frozen: -0.002 [-0.005, +0.001]; no indicators versus 14 families: -0.002, not significant. Every interval crosses zero.",
    "Line chart of yearly direction AUC of a logistic regression, 2017 to 2025, next-open fills: 0.527, 0.562, 0.561, 0.567, 0.553, 0.540, 0.551, 0.533, 0.518, with a 95% band, a dotted close-fill line and a dashed coin-flip line at 0.50. Three cards: network 0.521 versus logistic 0.570; 0.535 versus 0.552 on five judgement folds; forward test hit rate 50.3% over 11 unseen months.",
    "Seven translucent layers (news, GIS and terrain, elections, shareholder votes, finance and trade, social graph, physical events) stacked in perspective converge through a lens onto one thin line, the price. A callout says volatility predictability rose from 0.286 in 2018 to 0.399 in 2025 while direction skill faded.",
    "Diagram of the hyperspace prototype. A six-node geopolitical network on a faint graticule (teal alliances, red dashed competition, amber financial flow) sits above five blocks: temporal-pattern, semantic-embedding, structural-centrality, dynamic-agent, geospatial-kernel. They feed one 80-dimensional Universal Knowledge Tensor, then SVD kernels, sparse-autoencoder concepts, and a semantic canvas with governance flags.",
    "Closing slide: I do not know if it will work. I am optimistic. A list of planned next inputs: news, GIS layers, election results across democracies, shareholder votes across companies. A card links to github.com/another-worlds/hyperspace.",
]


# ================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    by = verify_facts()
    slides = [slide1(), slide2(), slide3(), slide4(), slide5(by), slide6(), slide7()]
    for k in ("hyper_nodes", "hyper_edges", "hyper_layers", "horizons"):
        _used.add(k)
    html_dir = HERE / "html"
    png_dir = HERE / "png"
    html_dir.mkdir(exist_ok=True)
    png_dir.mkdir(exist_ok=True)
    files = []
    for i, s in enumerate(slides, 1):
        p = html_dir / f"slide{i}.html"
        p.write_text(page(s), encoding="utf-8")
        files.append((p, png_dir / f"slide{i}.png", 1080, 1350))
    cp = html_dir / "cover.html"
    cp.write_text(cover(), encoding="utf-8")
    files.append((cp, png_dir / "cover_1200x627.png", 1200, 627))
    allp = HERE / "carousel.html"
    allp.write_text(page("".join(slides)), encoding="utf-8")
    (HERE / "alt_text.md").write_text("# Alt text (English)\n\n" + "\n\n".join(f"**Slide {i}.** {t}" for i, t in enumerate(ALT, 1)) +
                                      "\n\n**Cover (1200x627).** Same hero image as slide 1 with the headline: Do whales leave signatures in technical indicators?\n", encoding="utf-8")
    facts = [{"id": k, "value": FACTS[k][0], "shown_as": FACTS[k][1], "source_file": FACTS[k][2], "quote_or_key": FACTS[k][3]} for k in FACTS if k in _used]
    facts.append({"id": "recurrent_block", "value": "GRU", "shown_as": RECURRENT_LABEL, "source_file": "src/neural_trade/models/gru_attention.py",
                  "quote_or_key": "layers.Bidirectional(layers.GRU(64, return_sequences=True, ...)) ; the post says LSTM"})
    (HERE / "facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.no_render:
        return
    for src, out, w, h in files:
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--window-size={w},{h}", f"--screenshot={out}", src.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    pdf = HERE / "carousel.pdf"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", allp.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
