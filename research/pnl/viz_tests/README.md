Browser checks for the replay viewer (viz_template.html). Build the site first (viz_build.py), serve it
(cd <site> && python -m http.server 8765) with a test.html that wraps index.html in a full document, then run
from a scratch directory. Plotly is served locally because the CDN may be unreachable: every script needs
PLOTLY=<path to plotly-2.34.0.min.js>. PAGE=<page under http://localhost:8765/> picks another build (default test.html).

- phone_audit.js (DEVS=small,pixel7,iphone13): load and switch times, layout, CPU 4x, touch through CDP: vertical
  swipe scrolls the page, horizontal swipe pans, pinch zooms, tap fills the readout. Screenshots go to the cwd.
- desktop_check.js: data fidelity, time zones, readout, drag zoom, double-click, pin, strategy race, dark mode.
- regress_desktop.js, regress_touch.js, regress_layout.js, regress_data.js (SITE=<site dir> for ground truth):
  one check per confirmed finding of the first review round; each prints one JSON line per check and exits 1 on
  any failure.
- regress_r2_desktop.js, regress_r2_touch_data.js, regress_r2_layout.js: the same for the second review round
  (table while a strategy loads, readout height at 701-960 px, edge pins, double and triple clicks, pin loads vs
  hover, failed loads, held pans and pinches, card fit, landscape phones, panel titles and tick labels).
  regress_r2_touch_data.js judges phone readout heights with Roboto when ROBOTO=<Roboto woff2> is set (headless
  Chromium maps system-ui to the wider Inter font, so some wraps it shows do not happen on real phones).
  Checks marked advisory or info document a known residual and do not set the exit code.
- regress_r3_desktop_data.js, regress_r3_layout.js: the third and final review rounds (triple clicks with small moves,
  readout height at 701-720 px with a classic scrollbar, stalled 1-minute loads that fall back to the hourly close,
  renders during a drag zoom; short landscape charts, at least 2 labels per compact axis, distinct PnL labels on a
  flat PnL, the zoom-bar hint, the sealed-year label on landscape phones, height-only resizes). ROBOTO works as above.
  A triple click on a CPU 3-4x slower than the test machine needs real OS input (Xvfb + XTest) and is not scripted.
- regress_safe_area.js: iPhones with a notch or home indicator (CDP Emulation.setSafeAreaInsetsOverride, Chromium
  136+). Landscape and portrait cases with and without insets: no overflow, same readout height, the stuck readout
  does not cover the chart or the 'Sealed year' label, a tap pins. The published page uses viewport-fit=cover.
