"""Section screenshots and a console check of positive_outlook.html with headless Edge driven over the DevTools protocol (websocket-client).
Scrolls to each section like a reader, waits for its animations (real time), photographs the section, and records every console error / exception.
usage: python shoot_positive.py [--mode live|static] [--only id,id]   -> png_pos/<id>.png (live) or png_pos/static_<id>.png, plus mid-animation frames
Exit code 1 when any console error or exception was recorded."""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9341
SECS = [("top", 9.5), ("found", 3.5), ("clock", 6.5), ("geom", 6.0), ("barrier", 17.0), ("lines", 16.0), ("har", 11.0), ("reg", 8.0), ("next", 2.5), ("time", 5.0)]
MID = {"top": [2.0, 5.0], "clock": [0.5, 3.0], "barrier": [4.0, 9.0], "lines": [3.0, 6.5, 10.0], "har": [3.0, 6.0], "reg": [2.0], "geom": [1.5]}

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="live")
ap.add_argument("--only", default="")
ap.add_argument("--w", type=int, default=1500)
args = ap.parse_args()
out = os.path.join(HERE, "png_pos")
os.makedirs(out, exist_ok=True)
prof = tempfile.mkdtemp(prefix="edge_pos_")
proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={PORT}", "--user-data-dir=" + prof, "--disable-gpu", "--use-angle=swiftshader",
                         "--enable-unsafe-swiftshader", "--hide-scrollbars", "--no-first-run", f"--window-size={args.w},1000", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ws = None
for _ in range(60):
    try:
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
        page = [t for t in tabs if t["type"] == "page"][0]
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120, suppress_origin=True)
        break
    except Exception:
        time.sleep(0.5)
assert ws, "no devtools"
_id = 0
errors = []


def send(method, **params):
    global _id
    _id += 1
    ws.send(json.dumps({"id": _id, "method": method, "params": params}))
    while True:
        m = json.loads(ws.recv())
        if "method" in m:
            ev(m)
        if m.get("id") == _id:
            return m.get("result", m)


def ev(m):
    if m["method"] == "Runtime.exceptionThrown":
        d = m["params"]["exceptionDetails"]
        errors.append("EXC " + str(d.get("text")) + " " + str(d.get("exception", {}).get("description", ""))[:300])
    elif m["method"] == "Runtime.consoleAPICalled" and m["params"]["type"] in ("error", "warning"):
        errors.append("CONSOLE " + m["params"]["type"] + ": " + " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])[:300])
    elif m["method"] == "Log.entryAdded" and m["params"]["entry"]["level"] in ("error", "warning"):
        e = m["params"]["entry"]
        errors.append("LOG " + e["level"] + ": " + e["text"][:200] + " " + e.get("url", "")[-80:])


def pump(sec):
    """wait while draining events"""
    end = time.time() + sec
    ws.settimeout(0.2)
    while time.time() < end:
        try:
            m = json.loads(ws.recv())
            if "method" in m:
                ev(m)
        except websocket.WebSocketTimeoutException:
            pass
    ws.settimeout(120)


send("Runtime.enable")
send("Log.enable")
send("Page.enable")
send("Emulation.setDeviceMetricsOverride", width=args.w, height=1000, deviceScaleFactor=1, mobile=False)
url = "file:///" + os.path.join(HERE, "positive_outlook.html").replace("\\", "/") + ("?static=1" if args.mode == "static" else "")
send("Page.navigate", url=url)
pump(3.0)
only = [s for s in args.only.split(",") if s]


def ev_js(js):
    r = send("Runtime.evaluate", expression=js, returnByValue=True)
    return r.get("result", {}).get("value")


def shot(name, sec_id):
    rect = ev_js(f"(()=>{{const r=document.getElementById('{sec_id}').getBoundingClientRect();return [r.left+scrollX,r.top+scrollY,r.width,r.height]}})()")
    x, y, w, h = rect
    r = send("Page.captureScreenshot", format="png", captureBeyondViewport=True, clip={"x": x, "y": y, "width": w, "height": min(h, 6000), "scale": 1})
    p = os.path.join(out, name + ".png")
    open(p, "wb").write(base64.b64decode(r["data"]))
    print("shot", p, int(w), int(h), flush=True)


prefix = "static_" if args.mode == "static" else ""
for sid, wait in SECS:
    if only and sid not in only:
        continue
    ev_js(f"document.getElementById('{sid}').scrollIntoView({{block:'start'}})")
    t = 0.0
    if args.mode == "live":
        for m in MID.get(sid, []):
            pump(m - t)
            t = m
            if sid in ("top", "clock", "barrier", "har", "lines"):
                shot(f"mid_{sid}_{int(m * 10):03d}", sid)
        pump(max(0.2, wait - t))
    else:
        pump(2.0)
    shot(prefix + sid, sid)
    if sid == "found":
        pass
print("console/exception records:", len(errors))
for e in errors:
    print("  ", e)
try:
    send("Browser.close")
except Exception:
    pass
proc.terminate()
sys.exit(1 if errors else 0)
