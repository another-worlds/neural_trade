"""Screenshots of every section and every geometry block of tactical_report.html with headless Edge -> png/ (trimmed to the content)."""
import os, subprocess, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = "file:///" + os.path.join(HERE, "tactical_report.html").replace("\\", "/")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROF = os.path.join(os.environ.get("TEMP", "."), "edge_prof_tviz")
os.makedirs(os.path.join(HERE, "png"), exist_ok=True)
JOBS = [("s1", None), ("s2", None), ("s3", None)] + [("s4", b) for b in ("intro", "g_h24", "g_n1", "g_n2", "g_n3", "g_n4", "g_n5", "g_n6", "g_n7", "g_h67", "g_h80", "forest")] + \
       [("s5", None), ("s6", None), ("s7", None), ("s8", None)]


def shot(sec, blk, h=9000):
    name = f"{sec}" + (f"_{blk}" if blk else "")
    png = os.path.join(HERE, "png", name + ".png")
    if os.path.exists(png):
        os.remove(png)
    url = HTML + f"?only={sec}" + (f"&blk={blk}" if blk else "")
    subprocess.run([EDGE, "--headless=new", "--user-data-dir=" + PROF, "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars",
                    f"--screenshot={png}", f"--window-size=1500,{h}", "--virtual-time-budget=60000", url], capture_output=True, text=True)
    im = Image.open(png).convert("RGB")
    a = np.asarray(im)
    bg = np.array([0x12, 0x12, 0x11])
    rows = np.flatnonzero((np.abs(a.astype(int) - bg).sum(2) > 12).any(1))
    if len(rows):
        im = im.crop((0, 0, im.size[0], min(im.size[1], int(rows[-1]) + 30)))
    im.save(png)
    print(name, im.size, flush=True)


if __name__ == "__main__":
    only = sys.argv[1:]
    for sec, blk in JOBS:
        nm = sec + (f"_{blk}" if blk else "")
        if only and nm not in only and sec not in only:
            continue
        shot(sec, blk, 16000 if sec in ("s7", "s8") else 9000)
