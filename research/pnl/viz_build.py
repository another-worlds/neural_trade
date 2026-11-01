"""Build the PnL viewer page: inject meta, hourly price and the first strategy into viz_template.html.

  python viz_build.py SITE_DIR      (SITE_DIR from viz_export.py; writes SITE_DIR/index.html)
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(site):
    tpl = open(os.path.join(HERE, "viz_template.html")).read()
    meta = json.dumps(json.load(open(os.path.join(site, "meta.json"))), separators=(",", ":"))
    price = json.dumps(base64.b64encode(open(os.path.join(site, "price_hour.bin"), "rb").read()).decode())
    first = open(os.path.join(site, "s", "T1-LIN.json")).read()
    for blob in (meta, price, first):
        assert "</" not in blob
    html = tpl.replace("__META__", meta).replace("__PRICE__", price).replace("__FIRST__", first)
    open(os.path.join(site, "index.html"), "w").write(html)
    print(os.path.join(site, "index.html"), round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main(sys.argv[1])
