"""PNG of every slide of tactical_deck.html (headless Edge, 1600x900) -> png/NN.png. usage: python shoot_deck.py [n ...]"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = "file:///" + os.path.join(HERE, "tactical_deck.html").replace("\\", "/")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROF = os.path.join(os.environ.get("TEMP", "."), "edge_prof_deck")
os.makedirs(os.path.join(HERE, "png"), exist_ok=True)
N = 14
for n in ([int(a) for a in sys.argv[1:]] or range(1, N + 1)):
    png = os.path.join(HERE, "png", f"{n:02d}.png")
    subprocess.run([EDGE, "--headless=new", "--user-data-dir=" + PROF, "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars",
                    f"--screenshot={png}", "--window-size=1600,900", "--virtual-time-budget=20000", HTML + f"#{n}"], capture_output=True, text=True)
    print(n, os.path.exists(png), flush=True)
