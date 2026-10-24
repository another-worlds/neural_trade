"""Companion images for the LinkedIn post: clean, no technical specs. Reuses build_linkedin's visual system.

usage: python build_companion.py [--no-render]
Outputs html/ and PNGs: companion_1_five_blocks(.png|_square.png), companion_2_coin_flip(.png|_square.png).
Numbers are asserted by build_linkedin.verify_facts() (T09 yearly values, H82 hit rate, hyperspace graph).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_linkedin as B  # noqa: E402
from build_linkedin import (AMBER, BG0, DEFS, EDGE, GREEN, INDIGO, MINT, MONO, MUTED, MUTED2, RED, SKY, TXT,  # noqa: E402
                            card, esc, fact, network, page, tx)

W = 960
CHIPS = [("Markets", "How country markets are expected to move", SKY),
         ("News", "What the world is talking about", INDIGO),
         ("Alliances", "Who is aligned or in competition with whom", MINT),
         ("Simulated agents", "Bounded-rational actors trading resources and forming alliances", AMBER),
         ("Terrain and economy", "Elevation, climate, GDP, conflict: mapped to the same countries", GREEN)]


def extra(H):
    return (f".slide{{height:{H}px}} .c-h1{{font-size:60px;line-height:1.08;font-weight:800;letter-spacing:-.5px;color:{TXT}}}"
            f" .c-h1 em{{font-style:normal;color:{MINT}}} .c-h1 em.a{{color:{AMBER}}}")


def topbar(right=""):
    return (f'<div class="top"><span class="chip pr">prototype</span><span class="num">{right}</span></div>')


def image1(H):
    sq = H == 1080
    NH = 330 if sq else 420
    rh, rg = (64, 9) if sq else (84, 12)
    fs = 19 if sq else 21
    o = [f'<svg width="{W}" height="{NH}" viewBox="0 0 {W} {NH}" xmlns="http://www.w3.org/2000/svg">{DEFS}']
    o.append(card(0, 0, W, NH, MINT))
    o.append(f'<g transform="translate(0,12)">{network(W, NH - 22, pad=60 if sq else 66, rscale=24 if sq else 32, lsize=22 if sq else 25)}</g>')
    o.append(tx(22, 38, "bigger circle = more influence", 21, MUTED))
    lg = [("alliance", MINT, ""), ("competition", RED, "10 6"), ("financial flow", AMBER, "")]
    for i, (t, c, d) in enumerate(lg):
        yy = NH - 70 + i * 26
        da = f' stroke-dasharray="{d}"' if d else ""
        o.append(f'<line x1="{W-230}" y1="{yy-6}" x2="{W-190}" y2="{yy-6}" stroke="{c}" stroke-width="5"{da}/>')
        o.append(tx(W - 180, yy, t, 21, TXT))
    o.append("</svg>")
    rows = "".join(
        f'<div style="display:flex;align-items:center;height:{rh}px;margin-bottom:{rg}px;background:#0b1525;border:1px solid #1a2e50;'
        f'border-left:6px solid {c};border-radius:10px;padding:0 18px">'
        f'<div style="width:{256 if not sq else 236}px;flex:none;white-space:nowrap;font-size:{fs+3}px;font-weight:700;color:{c}">{esc(n)}</div>'
        f'<div style="font-size:{fs}px;line-height:1.25;color:{MUTED}">{esc(d)}</div></div>' for n, d, c in CHIPS)

    def pill(t, c):
        return (f'<div style="flex:1;text-align:center;white-space:nowrap;font-size:{27 if not sq else 24}px;font-weight:700;color:{c};background:#0b1525;'
                f'border:1px solid #1a2e50;border-radius:12px;padding:{16 if not sq else 12}px 6px">{t}</div>')
    arrow = f'<div style="font-size:34px;color:{MUTED};padding:0 12px">&#8594;</div>'
    flow = (f'<div style="display:flex;align-items:center;margin-top:{18 if not sq else 16}px">'
            + pill("one shared space", INDIGO) + arrow + pill("patterns found", AMBER) + arrow + pill("explained in plain language", MINT).replace("flex:1;", "flex:1.8;") + '</div>')
    h1sz = 52 if sq else 60
    inner = (topbar()
             + f'<div class="c-h1" style="margin-top:{14 if sq else 26}px;font-size:{h1sz}px">Five blocks of the world, one <em>shared</em> picture</div>'
             + f'<div style="margin-top:{24 if sq else 24}px">{"".join(o)}</div>'
             + f'<div style="margin-top:{12 if sq else 20}px">{rows}</div>' + flow)
    return f'<div class="slide">{inner}</div>'


def image2(H):
    sq = H == 1080
    r = json.loads((B.REPO / B.T09).read_text(encoding="utf-8"))["q1"]
    vals = [r["nopen"]["auc"][y]["mean"] for y in B.YEARS]
    assert " ".join(f"{v:.3f}" for v in vals) == B.FACTS["t09_nopen_auc"][1]
    B._used.update(["t09_nopen_auc", "h82_hit", "h82_months", "h82_trades", "nt104_net", "nt104_logreg", "h23_net", "h23_logreg"])
    CH = 570 if sq else 760
    L, R, T, Bm = 80, 50, 100, 60
    ymin, ymax = 0.49, 0.60
    px = lambda i: L + i * (W - L - R) / 8
    py = lambda v: T + (ymax - v) / (ymax - ymin) * (CH - T - Bm)
    o = [f'<svg width="{W}" height="{CH}" viewBox="0 0 {W} {CH}" xmlns="http://www.w3.org/2000/svg">{DEFS}']
    o.append(tx(0, 26, "How well it separates up-moves from down-moves", 26, TXT, weight=600))
    o.append(tx(0, 56, "0.50 = coin flip", 23, MUTED))
    for v in (0.50, 0.55, 0.60):
        o.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{W-R}" y2="{py(v):.1f}" stroke="#16263f" stroke-width="1.5"/>')
        o.append(tx(L - 14, py(v) + 8, f"{v:.2f}", 23, MUTED, "end", mono=True))
    o.append(f'<line x1="{L}" y1="{py(0.5):.1f}" x2="{W-R}" y2="{py(0.5):.1f}" stroke="{RED}" stroke-width="3.5" stroke-dasharray="10 7"/>')
    o.append(tx(L + 8, py(0.5) - 14, "coin flip", 24, RED, "start", 700))
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{AMBER}" stroke-width="7" stroke-linejoin="round" stroke-linecap="round"/>')
    peak = max(range(9), key=lambda i: vals[i])
    for i, v in enumerate(vals):
        big = i in (0, peak, 8)
        o.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="{10 if big else 6}" fill="{AMBER}" stroke="{BG0}" stroke-width="3"/>')
        if big:
            dy = -20 if i == peak else 38
            o.append(tx(px(i), py(v) + dy, f"{v:.3f}", 26, AMBER, "middle", 700, mono=True))
        o.append(tx(px(i), CH - 22, B.YEARS[i], 23, MUTED, "middle", mono=True))
    o.append("</svg>")
    c1 = (f'<div style="flex:1;background:#0b1525;border:1px solid #1a2e50;border-top:4px solid {AMBER};border-radius:12px;padding:{16 if not sq else 12}px 20px">'
          f'<div style="font-family:{MONO};font-size:{54 if not sq else 44}px;font-weight:700;color:{AMBER}">{fact("h82_hit")}</div>'
          f'<div style="font-size:{26 if not sq else 23}px;color:{TXT};margin-top:4px;line-height:1.3">of trades right on {fact("h82_months")}</div></div>')
    c2 = (f'<div style="flex:1;background:#0b1525;border:1px solid #1a2e50;border-top:4px solid {RED};border-radius:12px;padding:{16 if not sq else 12}px 20px;display:flex;align-items:center">'
          f'<div style="font-size:{31 if not sq else 27}px;font-weight:600;color:{TXT};line-height:1.3">The neural network scored lower than this simple model</div></div>')
    cap = f'<div style="margin-top:{16 if not sq else 10}px;font-size:22px;color:{MUTED2};text-align:right">BTC/USD, 1-minute data, 2017-2025</div>'
    inner = (f'<div style="height:44px"></div>'
             + f'<div class="c-h1" style="margin-top:{14 if sq else 26}px;font-size:{50 if sq else 60}px">Even the best simple model drifted to a <em class="a">coin flip</em></div>'
             + f'<div style="margin-top:{18 if sq else 24}px">{"".join(o)}</div>'
             + f'<div style="display:flex;gap:16px;margin-top:{10 if sq else 24}px">{c1}{c2}</div>' + cap)
    return f'<div class="slide">{inner}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    B.verify_facts()
    hd, out = HERE / "html", HERE
    hd.mkdir(exist_ok=True)
    jobs = []
    for name, fn in (("companion_1_five_blocks", image1), ("companion_2_coin_flip", image2)):
        for suf, H in (("", 1350), ("_square", 1080)):
            p = hd / f"{name}{suf}.html"
            p.write_text(page(fn(H), 1080, H, extra(H)), encoding="utf-8")
            jobs.append((p, out / f"{name}{suf}.png", H))
    if a.no_render:
        return
    for src, png, H in jobs:
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--window-size=1080,{H}", f"--screenshot={png}", src.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
