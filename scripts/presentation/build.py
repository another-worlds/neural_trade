"""Inline docs/presentation/data.json into the template -> docs/presentation/leader_c1.html (self-contained
apart from Plotly and Google Fonts, which load from their CDNs).

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


if __name__ == "__main__":
    main()
