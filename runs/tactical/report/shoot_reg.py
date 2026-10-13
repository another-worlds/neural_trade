"""Screenshots of every section of regression_report.html with headless Edge -> png_reg/ (trimmed to the content)."""
import os, subprocess, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = "file:///" + os.path.join(HERE, "regression_report.html").replace("\\", "/")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROF = os.path.join(os.environ.get("TEMP", "."), "edge_prof_reg")
OUT = os.path.join(HERE, "png_reg")
os.makedirs(OUT, exist_ok=True)
SECS = [f"s{i}" for i in range(12)]


def shot(sec, h=20000):
    png = os.path.join(OUT, sec + ".png")
    if os.path.exists(png):
        os.remove(png)
    subprocess.run([EDGE, "--headless=new", "--user-data-dir=" + PROF, "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars",
                    f"--screenshot={png}", f"--window-size=1500,{h}", "--virtual-time-budget=60000", HTML + f"?only={sec}"], capture_output=True, text=True)
    im = Image.open(png).convert("RGB")
    a = np.asarray(im)
    bg = np.array([0x12, 0x12, 0x11])
    rows = np.flatnonzero((np.abs(a.astype(int) - bg).sum(2) > 12).any(1))
    if len(rows):
        im = im.crop((0, 0, im.size[0], min(im.size[1], int(rows[-1]) + 30)))
    im.save(png)
    print(sec, im.size, flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or SECS):
        shot(s)
