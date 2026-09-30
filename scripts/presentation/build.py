"""Inline the extracted data (docs/presentation/*.json) into the templates -> presentations/1_leader_C1.html,
2_ensemble_C3.html, 3_ta_rule.html and 4_math_report.html (self-contained apart from Plotly and Google Fonts, which load from their CDNs).

    python scripts/presentation/build.py
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs/presentation"
OUT = REPO / "presentations"


def main() -> None:
    template = (DOCS / "template.html").read_text(encoding="utf-8")
    data = (DOCS / "data.json").read_text(encoding="utf-8").replace("</", "<\\/")
    OUT.mkdir(exist_ok=True)
    out = OUT / "1_leader_C1.html"
    out.write_text(template.replace("/*__DATA__*/", data), encoding="utf-8", newline="\n")
    print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
    candidate = (DOCS / "template_candidate.html").read_text(encoding="utf-8")
    for subject, title, name in (("ensemble", "Seed Ensemble C3", "2_ensemble_C3"), ("ta", "Gated Trend Rule", "3_ta_rule")):
        src = DOCS / f"data_{subject}.json"
        if not src.exists():
            continue
        page = candidate.replace("/*__TITLE__*/", title).replace(
            "/*__DATA__*/", src.read_text(encoding="utf-8").replace("</", "<\\/"))
        dst = OUT / f"{name}.html"
        dst.write_text(page, encoding="utf-8", newline="\n")
        print("wrote", dst, round(dst.stat().st_size / 1e6, 2), "MB")
    math = DOCS / "data_math.json"
    if math.exists():
        page = (DOCS / "template_math.html").read_text(encoding="utf-8").replace(
            "/*__DATA__*/", math.read_text(encoding="utf-8").replace("</", "<\\/"))
        dst = OUT / "4_math_report.html"
        dst.write_text(page, encoding="utf-8", newline="\n")
        print("wrote", dst, round(dst.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
