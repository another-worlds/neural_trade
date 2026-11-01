Browser checks for the replay viewer (viz_template.html). Build the site first (viz_build.py), serve it
(cd <site> && python -m http.server 8765) with a test.html that wraps index.html in a full document, then run
from a scratch directory: PLOTLY=<path to plotly-2.34.0.min.js> node phone_audit.js   (DEVS=small,pixel7,iphone13)
and node desktop_check.js. Plotly is served locally because the CDN may be unreachable. Screenshots go to the
current directory. phone_audit.js throttles the CPU 4x and drives touch through CDP: vertical swipe scrolls the
page, horizontal swipe pans, pinch zooms, tap fills the readout.
