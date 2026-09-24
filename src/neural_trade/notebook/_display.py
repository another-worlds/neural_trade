"""Put figures and tables INTO an ipywidgets Output, from any thread or a button callback.

``Output.append_display_data(fig)`` runs IPython's display formatter, and a Plotly figure has an
``_ipython_display_`` hook: formatting it *publishes* it to whichever cell is executing (the
``session.wait()`` cell during training, or nowhere when a button is clicked) instead of
returning data for the widget. ``Output.clear_output`` similarly talks to the executing cell.
Here the widget's ``outputs`` state is replaced directly with an explicit MIME bundle.
"""
from __future__ import annotations

import json


def mime_bundle(obj) -> dict:
    if hasattr(obj, "to_plotly_json"):  # plotly figure
        import plotly.io as pio

        # The plotly mime type only (what fig.show() publishes in Jupyter / VS Code): a text/html
        # fallback would embed the whole figure a second time and double the saved notebook.
        return {"application/vnd.plotly.v1+json": {**json.loads(pio.to_json(obj, validate=False)),
                                                   "config": {"responsive": True}},
                "text/plain": f"<Figure: {obj.layout.title.text or ''}>"}
    if hasattr(obj, "_repr_html_"):  # DataFrame / Styler
        return {"text/html": obj._repr_html_(), "text/plain": repr(obj)}
    return {"text/plain": str(obj)}


def show(output, obj) -> None:
    """Replace the contents of ``output`` with ``obj`` (state update only; safe from threads)."""
    output.outputs = ({"output_type": "display_data", "data": mime_bundle(obj), "metadata": {}},)
