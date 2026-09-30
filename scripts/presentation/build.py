"""Inline the extracted data into the templates -> docs/presentation/leader_c1.html, candidate_ensemble.html and
candidate_ta.html (self-contained apart from Plotly and Google Fonts, which load from their CDNs).

    python scripts/presentation/build.py
"""
from __future__ import annotations

from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs/presentation"


def main() -> None:
    template = (DOCS / "template.html").read_text(encoding="utf-8")
    data = (DOCS / "data.json").read_text(encoding="utf-8").replace("</", "<\\/")
    out = DOCS / "leader_c1.html"
    out.write_text(template.replace("/*__DATA__*/", data), encoding="utf-8", newline="\n")
    print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
    candidate = (DOCS / "template_candidate.html").read_text(encoding="utf-8")
    for subject, title in (("ensemble", "Seed Ensemble C3"), ("ta", "Gated Trend Rule")):
        src = DOCS / f"data_{subject}.json"
        if not src.exists():
            continue
        page = candidate.replace("/*__TITLE__*/", title).replace(
            "/*__DATA__*/", src.read_text(encoding="utf-8").replace("</", "<\\/"))
        dst = DOCS / f"candidate_{subject}.html"
        dst.write_text(page, encoding="utf-8", newline="\n")
        print("wrote", dst, round(dst.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
