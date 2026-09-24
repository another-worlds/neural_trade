"""Trading dashboard: one x axis, window-scoped numbers, trades at the window edges, hover text,
colour roles, the P(up) decision band, trade detail by view length, size budget."""
from __future__ import annotations

import re

import numpy as np
import pytest

from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")
_SPEC = re.compile(r"%\{[^}:|]*:([^}]*)\}")          # the d3 number format of a %{var:format} field
_D3 = re.compile(r"(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?")


def _fig(viz_backtest, **kw):
    from neural_trade.visualization.trading_dashboard import trading_dashboard_figure

    res, bars, sig, strat = viz_backtest
    return trading_dashboard_figure(res, bars, sig, strat, **kw)


def _trace(fig, name):
    (t,) = [t for t in fig.data if t.name == name]
    return t


def _y(t):
    return np.asarray(t.y, float)


def _xs(t):
    """x of a trace given as x0 / dx or as an array."""
    return np.asarray(t.x, float) if t.x is not None else t.x0 + t.dx * np.arange(len(t.y))


def _plotly_d3_ok(spec: str) -> bool:
    """plotly.js prepends '~' to a format not starting with one of ~ , . 0 $ that contains f/p/s/&
    (Lib.adjustFormat); d3-format must then still parse it, or plotly prints the raw number."""
    if not re.match(r"^[~,.0$]", spec) and re.search(r"[&fps]", spec):
        spec = "~" + spec
    return _D3.fullmatch(spec) is not None


def _viz_modules():
    from pathlib import Path

    import neural_trade.visualization as V

    return sorted(Path(V.__file__).parent.glob("*.py"))


# '+'-leading d3 formats still written in modules other groups own (finding #116); drop an entry once
# its module is fixed (the xfail is not strict, so a fixed module shows as XPASS until then)
_PENDING_D3 = {"trade_analytics.py", "indicator_evolution.py", "analytics_delta.py"}
_FIELD = re.compile(r"%\{+[^{}:|]*:([^{}]*)\}+")      # %{var:fmt} in a plain string or %{{var:fmt}} in an f-string
_FORMAT_KW = re.compile(r"(?:tick|hover|value|xhover|yhover)format\s*[=:]\s*[\"']([^\"']*)[\"']")


@pytest.mark.parametrize("path", [pytest.param(p, id=p.name, marks=[pytest.mark.xfail(
    reason="'+'-leading d3 format, finding #116 (another group's module)", strict=False)]
    if p.name in _PENDING_D3 else []) for p in _viz_modules()])
def test_every_visualization_module_writes_only_plotly_valid_number_formats(path):
    """plotly.js prints the raw number for a format d3 cannot parse (e.g. '+$,.2f' becomes '~+$,.2f'):
    scan the hover / text templates and the tick / hover formats written in every module."""
    bad = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for rx in (_FIELD, _FORMAT_KW):
            bad += [f"{path.name}:{i}: {spec!r}" for spec in rx.findall(line) if not _plotly_d3_ok(spec)]
    assert not bad, bad


def test_the_d3_check_rejects_what_plotly_rejects():
    assert all(_plotly_d3_ok(s) for s in ("$,.2f", ".3f", ".1%", ",.0f", "$,.0f", ".2%", "d", ".3s", "~s"))
    assert not any(_plotly_d3_ok(s) for s in ("+$,.2f", "+.3f", "+.2f", "+$,.0f"))
    assert _FIELD.findall('f"x %{{y:+.1f}}%"') == ["+.1f"] and _FIELD.findall('"%{y:$,.2f} %{x}"') == ["$,.2f"]


def test_trading_dashboard_shares_one_x_axis_and_marks_every_trade(viz_backtest):
    res, bars, sig, strat = viz_backtest
    fig = _fig(viz_backtest)
    assert {t.xaxis for t in fig.data} == {"x"} and fig.layout.hoversubplots == "axis"
    assert T.empty_panels(fig) == []
    entries = sum(len(t.x) for t in fig.data if t.name in ("long entry", "short entry"))
    exits = sum(len(t.x) for t in fig.data if t.yaxis == "y" and (t.name or "").startswith("exit, "))
    assert entries == exits == res.summary["n_trades"]
    window = _fig(viz_backtest, start=100, end=400)
    close = _trace(window, "close")
    assert len(close.y) == 300 and close.x0 == 100 and close.dx == 1


def test_every_panel_heading_is_drawn(viz_backtest):
    """A panel's heading is its legend's title; plotly draws no legend without a visible item."""
    for kw in ({}, dict(start=100, end=400)):
        fig = _fig(viz_backtest, **kw)
        legends = {t.legend for t in fig.data if t.showlegend is not False}
        panels = [k for k in fig.layout if k.startswith("yaxis")]
        assert len(legends) == len(panels) == 7, (legends, panels)


def test_window_subtitle_gives_the_window_result_and_the_whole_block_separately(viz_backtest):
    res, bars, sig, strat = viz_backtest
    lo, hi = 100, 400
    fig = _fig(viz_backtest, start=lo, end=hi)
    text = fig.layout.title.text
    E, G = np.asarray(res.equity), np.asarray(res.equity_gross)
    n_in = sum(lo <= t.entry_bar < hi for t in res.trades)
    assert f"window bars {lo}-{hi - 1}" in text and f": {n_in} trades entered" in text
    assert f"net {100 * (E[hi] / E[lo] - 1):+.2f}%" in text
    assert f"before costs {100 * (G[hi] - G[lo]) / E[lo]:+.2f}%" in text
    assert f"whole test block: {res.summary['n_trades']} trades" in text
    whole = _fig(viz_backtest).layout.title.text
    s = res.summary
    assert f"{s['n_trades']} trades · net {100 * s['total_return']:+.2f}%" in whole
    assert f"costs ${s['costs_paid']:,.0f}" in whole and f"max DD {100 * s['max_drawdown']:.2f}%" in whole


def test_window_pnl_restarts_at_the_window_and_drawdown_uses_the_window_peak(viz_backtest):
    res, bars, sig, strat = viz_backtest
    E, G, c = np.asarray(res.equity), np.asarray(res.equity_gross), np.asarray(bars.close)
    lo, hi = 700, 1300
    fig = _fig(viz_backtest, start=lo, end=hi)
    net, pre, bh, dd = (_y(_trace(fig, n)) for n in ("net P&L", "strategy before costs", "buy & hold (no costs)",
                                                      "drawdown"))
    np.testing.assert_allclose(net, E[lo + 1:hi + 1] - E[lo], atol=0.01)
    np.testing.assert_allclose(pre, G[lo + 1:hi + 1] - G[lo], atol=0.01)
    np.testing.assert_allclose(bh, E[lo] * (c[lo:hi] / c[lo - 1] - 1), atol=0.01)   # same base as the strategy
    peak = np.maximum.accumulate(E[lo:hi + 1])[1:]
    np.testing.assert_allclose(dd, E[lo + 1:hi + 1] / peak - 1, atol=1e-6)
    assert f"max DD {100 * -dd.min():.2f}%" in fig.layout.title.text
    # the whole block: buy & hold ends at the buy-and-hold baseline's P&L before costs
    full = _fig(viz_backtest)
    assert _y(_trace(full, "buy & hold (no costs)"))[-1] == pytest.approx(res.baselines["buy_and_hold"]["gross_pnl"],
                                                                           abs=0.05)
    assert _y(_trace(full, "net P&L"))[-1] == pytest.approx(E[-1] - E[0], abs=0.05)
    assert _y(_trace(full, "drawdown")).min() == pytest.approx(-res.summary["max_drawdown"], abs=1e-6)


def test_trades_crossing_the_window_edges_are_drawn_and_the_axis_stays_on_the_window(viz_backtest):
    res, bars, sig, strat = viz_backtest
    t = next(t for t in res.trades if t.bars_held >= 3 and t.entry_bar > 50)
    lo, hi = t.entry_bar + 1, t.entry_bar + 1 + 300
    fig = _fig(viz_backtest, start=lo, end=hi)
    assert tuple(fig.layout.xaxis.range) == (lo - 0.5, hi - 0.5) and fig.layout.xaxis.autorange is False
    exits = [x for tr in fig.data if tr.yaxis == "y" and (tr.name or "").startswith("exit, ") for x in tr.x]
    assert t.exit_bar in exits                                  # opened before the window, closed inside it
    for tr in fig.data:                                         # nothing is placed outside the window
        if tr.mode == "markers":
            assert np.all((_xs(tr) >= lo) & (_xs(tr) < hi)), tr.name
    rects = [s for s in fig.layout.shapes if s.type == "rect" and s.xref == "x"]
    assert any(s.x0 < lo <= s.x1 for s in rects)                # its holding period reaches the left edge
    assert "1 trade open at the window start" in fig.layout.title.text
    gap = next((a, b) for a, b in zip(res.trades, res.trades[1:]) if b.entry_bar - a.exit_bar > 8)
    empty = _fig(viz_backtest, start=gap[0].exit_bar + 1, end=gap[1].entry_bar)
    assert "no trades in this view" in empty.layout.legend.title.text and T.empty_panels(empty) == []


def test_hover_formats_are_valid_and_levels_are_preformatted(viz_backtest):
    res, bars, sig, strat = viz_backtest
    for kw in ({}, dict(start=100, end=400)):
        fig = _fig(viz_backtest, **kw)
        specs = [s for t in fig.data for s in _SPEC.findall(t.hovertemplate or "")]
        assert specs and all(_plotly_d3_ok(s) for s in specs), specs
        assert not any(t.customdata is not None and "TP" in (t.hovertemplate or "") for t in fig.data)
    assert all(t.tp is None for t in res.trades)                 # calibrated_quantile sets a stop only
    lo, hi = 100, 400
    fig = _fig(viz_backtest, start=lo, end=hi)
    assert "take-profit" not in fig.layout.legend.title.text and "stop levels" in fig.layout.legend.title.text
    text = _trace(fig, "close").text
    t = next(t for t in res.trades if lo <= t.entry_bar < hi)
    entry = text[t.entry_bar - lo]
    assert f"{t.side} entry" in entry and f"stop ${t.sl:,.2f}" in entry and "take-profit" not in entry
    assert "$0.00" not in "".join(text)


def test_the_unified_hover_lists_an_event_only_on_its_own_bar(viz_backtest):
    res, bars, sig, strat = viz_backtest
    lo, hi = 100, 400
    fig = _fig(viz_backtest, start=lo, end=hi)
    for tr in fig.data:                                    # markers do not answer a hover at a nearby bar
        if tr.mode == "markers":
            assert tr.hoverinfo == "skip", tr.name
    text = _trace(fig, "close").text
    t = next(t for t in res.trades if lo <= t.entry_bar and t.exit_bar < hi)
    ex = text[t.exit_bar - lo]
    assert f"{t.side} exit ({t.exit_reason})" in ex and f"entered bar {t.entry_bar}" in ex
    assert f"net {'+' if t.net_pnl > 0 else '-'}${abs(t.net_pnl):,.2f}" in ex
    assert "decided" in text[t.entry_bar - 1 - lo]         # the decision is at the close before the fill
    events = {i for i, s in enumerate(text) if s}
    expected = {b - lo for tr in res.trades for b in (tr.entry_bar, tr.exit_bar) if lo <= b < hi}
    expected |= {d["bar"] - lo for d in res.decisions if lo <= d["bar"] < hi}
    assert events == expected


def test_colours_follow_roles(viz_backtest):
    fig = _fig(viz_backtest, start=100, end=400)
    horizon = {c.lower() for c in T.HORIZON_COLORS.values()}

    def base(c):
        if c is None:
            return None
        m = re.match(r"rgba\((\d+),(\d+),(\d+)", c.replace(" ", ""))
        return "#%02x%02x%02x" % tuple(int(v) for v in m.groups()) if m else c.lower()

    for tr in fig.data:
        line = base(tr.line.color) if tr.line is not None else None
        mark = base(tr.marker.color) if tr.marker is not None and isinstance(tr.marker.color, str) else None
        if tr.name in H or tr.name == "sigma h1":
            assert line == T.HORIZON_COLORS[tr.name.split()[-1]]
            continue
        assert line not in horizon and mark not in horizon, tr.name          # horizon colours mean horizons
        if tr.name in ("long entry", "short entry"):
            assert mark == (T.LONG_COLOR if tr.name.startswith("long") else T.SHORT_COLOR).lower()
            assert tr.marker.symbol == ("triangle-up" if tr.name.startswith("long") else "triangle-down")
        elif (tr.name or "").startswith("exit, "):
            assert tr.marker.symbol in ("circle", "x") and mark in (T.GOOD.lower(), T.CRITICAL.lower())
        elif "(entry to exit)" in tr.name:                                  # outcome lines: toggled with the exit key
            assert tr.showlegend is False and tr.legendgroup in ("trade-win", "trade-loss")
        elif tr.mode == "lines" and tr.name != "drawdown":
            assert line not in (T.LONG_COLOR.lower(), T.SHORT_COLOR.lower(), T.GOOD.lower(), T.CRITICAL.lower()), \
                tr.name                                                     # e.g. signal strength, stop level
        if tr.line is not None and tr.line.dash is not None:
            assert tr.line.dash != T.TRAIN_DASH, tr.name                    # dotted means training
    for s in fig.layout.shapes:                                             # the strategy's lines
        if s.type == "line":
            assert base(s.line.color) not in horizon | {T.LONG_COLOR.lower(), T.SHORT_COLOR.lower()}
            assert s.line.dash != T.TRAIN_DASH


def test_the_pup_axis_fits_the_decision_band_and_marks_the_entry_zones(viz_backtest):
    res, bars, sig, strat = viz_backtest
    lo, hi = 100, 400
    fig = _fig(viz_backtest, start=lo, end=hi)
    y0, y1 = fig.layout.yaxis2.range
    w = np.asarray(sig.weighted_direction)[lo:hi]
    assert y0 < min(strat.short_below, w.min()) and y1 > max(strat.long_above, w.max())
    assert y1 - y0 < 0.9 * (np.asarray(sig.p)[lo:hi].max() - np.asarray(sig.p)[lo:hi].min()) + 0.1
    zones = [s for s in fig.layout.shapes if s.type == "rect" and s.yref == "y2"]
    assert sorted(round(min(s.y0, s.y1), 6) for s in zones) == sorted(
        [0.0, round(strat.long_above, 6)])
    labels = [a.text for a in fig.layout.annotations if a.yref == "y2"]
    assert any(lab.startswith("long >") for lab in labels) and any(lab.startswith("short <") for lab in labels)
    dec = [tr for tr in fig.data if tr.name in ("decided long", "decided short")]
    xs = sorted(x for tr in dec for x in tr.x)
    assert xs == sorted(d["bar"] for d in res.decisions if lo <= d["bar"] < hi)
    for tr in dec:
        np.testing.assert_allclose(tr.y, np.asarray(sig.weighted_direction)[list(tr.x)], atol=1e-6)
        side = tr.name.split()[-1]                         # each side has its own key, colour and triangle
        assert tr.showlegend is not False and tr.legendgroup == f"decision-{side.upper()}"
        assert tr.marker.symbol == ("triangle-up" if side == "long" else "triangle-down")
        assert tr.marker.color == (T.LONG_COLOR if side == "long" else T.SHORT_COLOR)
    assert {tr.name for tr in dec} == {"decided long", "decided short"}


def test_one_horizon_spike_does_not_squeeze_the_decision_band(viz_backtest):
    """A multi-bar spike of one horizon widens the P(up) axis by at most _PUP_REACH + _PUP_PAD beyond
    the traded series and the lines; the points left outside are counted in the subtitle."""
    import dataclasses

    from neural_trade.visualization.trading_dashboard import _PUP_PAD, _PUP_REACH, trading_dashboard_figure

    res, bars, sig, strat = viz_backtest
    p = np.array(sig.p, float)
    p[1500:1520, 1] = 0.97                                  # h1 pinned high for 20 bars
    p[1530:1535, 0] = 0.03
    spiky = dataclasses.replace(sig, p=p)
    for lo, hi in ((1400, 1700), (1000, 2000), (0, len(bars))):
        fig = trading_dashboard_figure(res, bars, spiky, strat, start=lo, end=hi)
        y0, y1 = fig.layout.yaxis2.range
        w = np.asarray(sig.weighted_direction)[lo:hi]
        core_lo = min(w.min(), strat.short_below, strat.long_above, strat.median)
        core_hi = max(w.max(), strat.short_below, strat.long_above, strat.median)
        assert y0 < core_lo and y1 > core_hi                          # the traded series and lines stay in
        assert y0 >= core_lo - _PUP_REACH - _PUP_PAD - 1e-9 and y1 <= core_hi + _PUP_REACH + _PUP_PAD + 1e-9
        assert (strat.long_above - strat.short_below) / (y1 - y0) > 0.15
        out = int(np.sum((p[lo:hi] < y0) | (p[lo:hi] > y1)))
        assert out >= 25
        assert f"the P(up) axis leaves out {out} of {p[lo:hi].size:,} horizon points" in fig.layout.title.text


def test_the_horizon_range_band_is_named_for_what_it_aggregates(viz_backtest):
    """Views of 801-1500 bars draw the per-bar min-max across the horizons (no bucket); longer views
    bucket the bars too, and the legend and hover say so."""
    fig = _fig(viz_backtest, start=1000, end=2000)                 # 1000 bars: not detail, 1 bar per point
    (band,) = [t for t in fig.data if (t.name or "").startswith("h0-h2 range")]
    assert band.name == "h0-h2 range (min-max across horizons)" and band.dx == 1
    templates = " ".join(t.hovertemplate for t in fig.data if t.legendgroup == "h-range")
    assert "h0-h2 max %{y:.3f}" in templates and "h0-h2 min %{y:.3f}" in templates
    assert "over 1 bars" not in templates and "per 1 bars" not in band.name
    full = _fig(viz_backtest)                                      # 3000 bars: 2-bar buckets
    (band,) = [t for t in full.data if (t.name or "").startswith("h0-h2 range")]
    assert band.name == "h0-h2 range (min-max across horizons and 2-bar buckets)" and band.dx == 2
    assert any("h0-h2 max over 2 bars" in (t.hovertemplate or "") for t in full.data)


def _other(viz_backtest, name, params=None):
    from neural_trade.strategy import backtest, build_backtest_config, build_strategy

    res, bars, sig, _ = viz_backtest
    strat = build_strategy(name, params, calibration=sig)
    return backtest(sig, bars, strat, build_backtest_config({"random_seeds": 0})), bars, sig, strat


def test_a_strategy_without_trades_reads_cleanly(viz_backtest):
    from neural_trade.visualization.trading_dashboard import _MIN_DD_SPAN, _MIN_USD_SPAN

    flat = _other(viz_backtest, "always_flat")
    assert flat[0].summary["n_trades"] == 0
    for kw in ({}, dict(start=2400, end=3000)):
        fig = _fig(flat, **kw)
        text = fig.layout.title.text
        assert "max DD 0.00%" in text and "-0.00" not in text and "+0.00%" not in text
        assert "0 trades" in text and "0 of 0" not in text and "won after costs" not in text
        lo_, hi_ = fig.layout.yaxis5.autorangeoptions.include       # net P&L: whole, distinct $ ticks
        assert hi_ - lo_ == pytest.approx(_MIN_USD_SPAN) and lo_ < 0 < hi_
        y0, y1 = fig.layout.yaxis7.range                           # drawdown: 0 to about -1%, not +/-100%
        assert -1.1 * _MIN_DD_SPAN < y0 < -_MIN_DD_SPAN / 2 and 0 < y1 < 0.1 * _MIN_DD_SPAN
    moving = _fig(viz_backtest, start=100, end=400)               # a real drawdown keeps its autorange
    assert moving.layout.yaxis7.range is None and moving.layout.yaxis5.autorangeoptions.include is None


def test_counts_are_singular_for_one_and_the_window_note_is_on_the_window_line(viz_backtest):
    bh = _other(viz_backtest, "buy_and_hold")
    assert bh[0].summary["n_trades"] == 1
    text = _fig(bh).layout.title.text
    assert ": 1 trade ·" in text and "1 trades" not in text and " of 1 trade won after costs" in text
    lines = _fig(viz_backtest, start=100, end=400).layout.title.text.split("<br>")
    window = next(i for i, ln in enumerate(lines) if "window bars 100-399" in ln)
    note = next(i for i, ln in enumerate(lines) if "% are of the window's starting equity" in ln)
    whole = next(i for i, ln in enumerate(lines) if "whole test block" in ln)
    assert window <= note < whole                                     # with the window's numbers
    assert "starting equity" not in lines[whole]


def test_take_profit_and_stop_levels_are_told_apart(viz_backtest):
    ts = _other(viz_backtest, "threshold_spike", {"p_long": 0.55, "p_short": 0.45, "min_confidence": 0.0})
    res, bars = ts[0], ts[1]
    assert all(np.isfinite(t.tp) and np.isfinite(t.sl) for t in res.trades)
    both = 0
    for t0 in res.trades[5::max(1, len(res.trades) // 6)][:6]:
        lo, hi = max(0, t0.entry_bar - 150), min(len(bars), t0.entry_bar + 150)
        fig = _fig(ts, start=lo, end=hi)
        tp, sl = _trace(fig, "take-profit level"), _trace(fig, "stop level")
        assert tp.line.dash != sl.line.dash and tp.line.color != sl.line.color
        assert T.TRAIN_DASH not in (tp.line.dash, sl.line.dash)   # dotted means training
        top, bottom = np.max(bars.high[lo:hi]), np.min(bars.low[lo:hi])
        pad = 0.05 * (top - bottom)                              # the price axis: the bars plus 5%
        trades = [t for t in res.trades if t.exit_bar >= lo and t.entry_bar < hi]
        drawn = 0
        for level, line, tag in (("tp", tp, "TP"), ("sl", sl, "stop")):
            want = sorted((max(t.entry_bar, lo), getattr(t, level) > t.entry_price) for t in trades
                          if bottom - pad <= getattr(t, level) <= top + pad)       # levels on the axis
            labs = [tr for tr in fig.data if tr.name == f"{line.name} label"]
            if not want:
                assert not labs
                continue
            (lab,) = labs
            drawn += 1
            assert set(lab.text) == {tag} and lab.legendgroup == line.legendgroup and lab.showlegend is False
            assert lab.hoverinfo == "skip" and min(lab.x) >= lo
            got = sorted((x, pos == "bottom right") for x, pos in zip(lab.x, lab.textposition))
            assert got == want                                   # on its segment, facing the entry price
        both += drawn == 2
    assert both
    only_stop = _fig(viz_backtest, start=100, end=400)          # calibrated_quantile: stops only, no labels
    assert not [t for t in only_stop.data if (t.name or "").endswith("level label")]


def test_long_views_leave_out_trade_detail_and_say_so(viz_backtest):
    res, bars, sig, strat = viz_backtest
    full = _fig(viz_backtest)                                      # 3000 bars > DETAIL_MAX_BARS
    names = {t.name for t in full.data}
    assert "stop level" not in names and not any("(entry to exit)" in (n or "") for n in names)
    assert not [s for s in full.layout.shapes if s.type == "rect" and s.xref == "x"]
    assert "drawn on views of 800 bars or fewer" in full.layout.title.text.replace("<br>", " ")
    assert not set(H) & names and any((n or "").startswith("h0-h2 range") for n in names)
    win = _fig(viz_backtest, start=100, end=400)
    names = {t.name for t in win.data}
    assert "stop level" in names and set(H) <= names
    rects = [s for s in win.layout.shapes if s.type == "rect" and s.xref == "x"]
    assert len(rects) == sum(t.exit_bar >= 100 and t.entry_bar < 400 for t in res.trades)


def test_size_budget_x0_dx_and_float32(viz_backtest):
    res, bars, sig, strat = viz_backtest
    fig = _fig(viz_backtest)
    n = len(bars)
    for tr in fig.data:
        if tr.y is not None and len(tr.y) >= n // 4:
            assert tr.x is None and tr.dx is not None, tr.name              # no repeated arange x arrays
            assert np.asarray(tr.y).dtype == np.float32, tr.name
    # ~600 kB for a 7,236-bar block, plus the per-trade markers and hover lines (this fixture trades
    # every ~7 bars; the real run every ~44)
    assert len(fig.to_json()) < 600_000 * n / 7236 + 200 * res.summary["n_trades"]


def test_registry_entry_passes_the_config_through(viz_backtest, viz_config):
    from neural_trade.registries.visualizations import Visualizations

    res, bars, sig, strat = viz_backtest
    fig = Visualizations.build("trading_dashboard", res, viz_config, bars=bars, signals=sig, strategy=strat,
                               start=100, end=400)
    label = T.horizon_label("h1", viz_config)
    assert label != "h1" and any(label in (t.hovertemplate or "") for t in fig.data)


def test_detail_window_picks_a_readable_window(viz_backtest):
    from neural_trade.visualization.trading_dashboard import detail_window

    res = viz_backtest[0]
    n, E = len(res.position), np.asarray(res.equity)
    s, e = detail_window(res, 500)
    assert e - s == 500 and 0 <= s and e <= n
    assert E[e] - E[s] == pytest.approx(min(E[i + 500] - E[i] for i in range(n - 500 + 1)))
    assert detail_window(res, 500, around="last") == (n - 500, n)
    worst = min(res.trades, key=lambda t: t.net_pnl)
    s, e = detail_window(res, 400, around="worst_trade")
    assert s <= worst.entry_bar < e and e - s == 400
    with pytest.raises(ValueError):
        detail_window(res, 400, around="best")
    with pytest.raises(ValueError):
        _fig(viz_backtest, start=500, end=500)
