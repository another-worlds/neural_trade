Browser checks for the replay viewer (viz_template.html). Build the site first (viz_build.py), serve it
(cd <site> && python -m http.server 8765) with a test.html that wraps index.html in a full document, then run
from a scratch directory. Plotly is served locally because the CDN may be unreachable: every script needs
PLOTLY=<path to plotly-2.34.0.min.js>. PAGE=<page under http://localhost:8765/> picks another build (default test.html).

- phone_audit.js (DEVS=small,pixel7,iphone13): load and switch times, layout, CPU 4x, touch through CDP: vertical
  swipe scrolls the page, horizontal swipe pans, pinch zooms, tap fills the readout. Screenshots go to the cwd.
- desktop_check.js: data fidelity, time zones, readout, drag zoom, double-click, pin, strategy race, dark mode.
- regress_desktop.js, regress_touch.js, regress_layout.js, regress_data.js (SITE=<site dir> for ground truth):
  one check per confirmed review finding; each prints one JSON line per check and exits 1 on any failure.
