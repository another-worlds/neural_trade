"""Per-trade analytics and the strategy comparison."""
from __future__ import annotations

import json

import numpy as np
import pytest

from neural_trade.visualization import theme as T

SLIP = 3e-4   # the engine's default half-spread + slippage (1 + 2 bps)


def _bars(rows):
    from neural_trade.strategy import Bars

    o, h, lo, c = (np.array(col, dtype=float) for col in zip(*rows))
    return Bars(o, h, lo, c)


def _trade(side, entry_bar, exit_bar, entry_mid, exit_mid, reason):
    from neural_trade.strategy import Trade

    s = 1 if side == "LONG" else -1
    return Trade(side=side, entry_bar=entry_bar, exit_bar=exit_bar, entry_price=entry_mid * (1 + s * SLIP),
                 exit_price=exit_mid * (1 - s * SLIP), notional=10_000.0, gross_pnl=s * 100 * (exit_mid - entry_mid),
                 costs=26.0, exit_reason=reason)


# ------------------------------------------------------------------ excursions (MFE / MAE)
def test_excursions_window_follows_the_engine_fills():
    """Hand-computed windows: the entry bar counts from its open; a next-open exit counts only
    the exit bar's open; an intrabar stop counts the stop, not the rest of the exit bar; EOW
    counts the whole last bar. MFE / MAE are in % of the entry mid (the entry bar's open)."""
    from neural_trade.visualization.trade_analytics import excursions

    bars = _bars([
        (100, 100, 100, 100),   # 0  decision bar
        (100, 103, 98, 101),    # 1  long enters at this OPEN: its low 98 is the worst move
        (101, 104, 99, 102),    # 2  held
        (101, 150, 50, 120),    # 3  REV exit at this OPEN (101): the spike after it must not count
        (200, 200, 200, 200),   # 4
        (200, 201, 199, 200),   # 5  short enters at the open (200)
        (200.5, 210, 190, 205),  # 6  stopped at 203 inside this bar: 210 / 190 may come after the stop
        (100, 100, 100, 100),   # 7
        (100, 102, 97, 99),     # 8  long enters, held to the end of the block
        (99, 106, 95, 105),     # 9  EOW: closed at this bar's CLOSE, so the whole bar counts
        (100, 100, 90, 95),     # 10 long enters and is stopped at 99 in the same bar
    ])
    trades = [_trade("LONG", 1, 3, 100, 101, "REV"), _trade("SHORT", 5, 6, 200, 203, "SL"),
              _trade("LONG", 8, 9, 100, 105, "EOW"), _trade("LONG", 10, 10, 100, 99, "SL")]
    mfe, mae = excursions(trades, bars)     # the exit mid is recovered from each trade's own entry fill
    np.testing.assert_allclose(mfe, [4.0, 0.5, 6.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(mae, [-2.0, -1.5, -5.0, -1.0], atol=1e-9)
    mfe2, mae2 = excursions(trades, bars, slip_rate=SLIP)
    np.testing.assert_allclose(mfe2, mfe, atol=1e-9)
    np.testing.assert_allclose(mae2, mae, atol=1e-9)


def test_excursions_ignore_the_bar_after_a_next_open_exit_in_a_real_backtest():
    """Engine round trip: a TIME exit fills at the next open, so a spike in that bar changes the
    trade's P&L not at all and its MFE not at all."""
    from neural_trade.strategy import BacktestConfig, Order, Strategy, run_backtest
    from neural_trade.visualization.trade_analytics import excursions

    class LongOnce(Strategy):
        name = "long_once"

        def decide(self, s, t):
            return Order("LONG", 1.0, max_hold=2, reason="test") if t == 0 else None

    def run(spike):
        rows = [(100, 100.5, 99.5, 100), (100, 101, 99, 100.5), (100.5, 102, 100, 101),
                (101, 101 + spike, 100.5, 101), (101, 101.5, 100.5, 101)]
        bars = _bars(rows)
        sig = type("S", (), {"__len__": lambda self: len(rows)})()
        res = run_backtest(sig, bars, LongOnce(), BacktestConfig(random_seeds=0))
        return res, bars

    (r0, b0), (r1, b1) = run(0.5), run(500.0)
    assert [t.exit_reason for t in r0.trades] == ["TIME"] and r0.trades[0].exit_bar == 3
    assert r0.trades[0].gross_pnl == pytest.approx(r1.trades[0].gross_pnl)
    m0, a0 = excursions(r0.trades, b0)
    m1, a1 = excursions(r1.trades, b1)
    np.testing.assert_allclose(m0, [2.0], atol=1e-9)        # bar 2's high 102 vs the entry mid 100
    np.testing.assert_allclose(a0, [-1.0], atol=1e-9)       # bar 1's low 99 (the entry bar itself)
    np.testing.assert_allclose(m1, m0, atol=1e-12)
    np.testing.assert_allclose(a1, a0, atol=1e-12)


def test_excursions_bound_the_realised_move_and_every_winner_passed_break_even(viz_backtest):
    from neural_trade.visualization.trade_analytics import break_even_pct, excursions

    res, bars, _, _ = viz_backtest
    mfe, mae = excursions(res.trades, bars)
    gross_pct = np.array([100 * t.gross_pnl / t.notional for t in res.trades])
    assert len(mfe) == len(res.trades)
    assert np.all(mae <= 1e-12) and np.all(mfe >= -1e-12)          # the entry mid is inside the window
    assert np.all(mae <= gross_pct + 1e-9) and np.all(gross_pct <= mfe + 1e-9)
    for t, f in zip(res.trades, mfe):
        if t.net_pnl > 0:
            assert f >= break_even_pct(res.config, t.side) - 1e-9
    # exact break-even on mids: 2c / (1 - c) for a long with c = 13 bps per side
    assert break_even_pct(res.config, "LONG") == pytest.approx(100 * 2 * 0.0013 / (1 - 0.0013))


# ------------------------------------------------------------------ the per-trade figure
def _traces(fig, xaxis):
    return [t for t in fig.data if (getattr(t, "xaxis", None) or "x") == xaxis]


def test_trade_analytics_draws_every_panel_with_its_own_key(viz_backtest):
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    fig = trade_analytics_figure(res, bars, signals=sig)
    assert T.empty_panels(fig) == []
    # one key per panel, positioned over that panel; no shared legend row across the figure
    keyed = {t.legend or "legend" for t in fig.data if t.showlegend is not False}
    assert keyed == {"legend"} | {f"legend{i}" for i in range(2, 10)}
    for i in range(1, 10):
        leg = fig.layout["legend" if i == 1 else f"legend{i}"]
        sp = fig.get_subplot((i - 1) // 3 + 1, (i - 1) % 3 + 1)
        assert leg.x == pytest.approx(sp.xaxis.domain[0]) and leg.y == pytest.approx(sp.yaxis.domain[1])
    # reference lines are labelled in the keys, not by text on top of the data: the only
    # annotations are the nine panel titles (each with its one-line readout)
    assert len(fig.layout.annotations) == 9 and all("<br>" in a.text for a in fig.layout.annotations)
    assert len(fig.to_json()) < 600_000


def test_trade_analytics_colours_by_role(viz_backtest):
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    fig = trade_analytics_figure(res, bars, signals=sig)
    js = json.dumps(json.loads(fig.to_json())["data"]).lower()     # the traces (the template lists every slot)
    for c in T.HORIZON_COLORS.values():             # no horizon colours in a per-trade figure
        assert c.lower() not in js
    for t in fig.data:                              # status colours only with a win / loss shape
        m = getattr(t, "marker", None)
        color = getattr(m, "color", None)
        if color in (T.GOOD, T.CRITICAL):
            assert getattr(m, "symbol", None) == ("circle" if color == T.GOOD else "x"), t.name
    # exit-reason boxes are neutral; outcome lives in the points
    boxes = [t for t in fig.data if t.type == "box" and t.hoveron == "boxes"]
    assert boxes and all(b.line.color == T.INK_2 for b in boxes)
    assert all("won" in b.x[0] for b in boxes)       # category label: count, winners, net $
    # the cumulative-costs line is a series, not the status red
    costs = [t for t in fig.data if t.name == "costs"]
    assert costs and costs[0].line.color not in (T.CRITICAL, T.GOOD)


def test_trade_analytics_per_trade_panels_are_in_percent_of_notional(viz_backtest):
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    fig = trade_analytics_figure(res, bars, signals=sig)
    gross_pct = np.array([100 * t.gross_pnl / t.notional for t in res.trades])
    net_pct = np.array([100 * t.net_pnl / t.notional for t in res.trades])
    scat = [t for t in _traces(fig, "x2") if t.mode == "markers"][0]
    np.testing.assert_allclose(np.asarray(scat.x, float), gross_pct, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(scat.y, float), net_pct, rtol=1e-5, atol=1e-5)
    # the $ values stay in the hover
    np.testing.assert_allclose(np.asarray(scat.customdata, float)[:, 1], [t.net_pnl for t in res.trades], rtol=1e-5)
    # every trade is in the histogram, won and lost never share a bin (0 is a bin edge)
    hist = [t for t in _traces(fig, "x") if t.type == "bar"]
    assert sum(int(np.sum(t.y)) for t in hist) == len(res.trades)
    both = (np.asarray(hist[0].y) > 0) & (np.asarray(hist[1].y) > 0)
    assert not both.any()
    # cumulative P&L stays in $ (summed $ P&L is the equity change)
    net_line = [t for t in _traces(fig, "x4") if t.name == "net"][0]
    assert float(np.asarray(net_line.y)[-1]) == pytest.approx(sum(t.net_pnl for t in res.trades), rel=1e-5)


def test_trade_analytics_holding_time_is_jittered_but_hovers_true_bars(viz_backtest):
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    fig = trade_analytics_figure(res, bars, signals=sig)
    pts = [t for t in _traces(fig, "x5") if t.mode == "markers"]
    assert pts
    for t in pts:
        held = np.asarray(t.customdata, float)[:, 0]
        x = np.asarray(t.x, float)
        assert np.all(held == np.round(held)) and np.all(np.abs(x - held) <= 0.3 + 1e-6)
        assert "customdata[0]" in t.hovertemplate and "%{x}" not in t.hovertemplate
    assert sum(len(t.x) for t in pts) == len(res.trades)


def test_trade_analytics_reads_the_signal_at_the_decision_bar(viz_backtest):
    """The order fills at the next open, so the entry signal is the one at entry_bar - 1."""
    from neural_trade.visualization import stats as S
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    fig = trade_analytics_figure(res, bars, signals=sig)
    tr = res.trades
    sign = np.array([1 if t.side == "LONG" else -1 for t in tr])
    gross_pct = np.array([100 * t.gross_pnl / t.notional for t in tr])

    def quintile_means(bar_of):
        conv = sign * (sig.weighted_direction[[bar_of(t) for t in tr]] - 0.5)
        k = int(min(5, max(1, len(tr) // 8)))
        qs = np.quantile(conv, np.linspace(0, 1, k + 1))
        b = np.clip(np.searchsorted(qs[1:-1], conv, side="right"), 0, k - 1)
        return [S.mean_ci(gross_pct[b == j])[0] for j in range(k) if (b == j).any()]

    line = [t for t in _traces(fig, "x7") if t.error_y is not None and t.error_y.array is not None][0]
    np.testing.assert_allclose(np.asarray(line.y, float), quintile_means(lambda t: t.entry_bar - 1), rtol=1e-9)
    assert not np.allclose(np.asarray(line.y, float), quintile_means(lambda t: t.entry_bar))
    # predicted move: the h1 delta at the decision bar, in the trade's direction
    pred = np.concatenate([np.asarray(t.x, float) for t in _traces(fig, "x9") if t.mode == "markers"])
    want = np.concatenate([(1 if s == "LONG" else -1) * sig.delta[[t.entry_bar - 1 for t in tr if t.side == s], 1]
                           for s in ("LONG", "SHORT")])
    np.testing.assert_allclose(pred, want, rtol=1e-5, atol=1e-4)


def test_trade_analytics_long_vs_short_totals(viz_backtest):
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    fig = trade_analytics_figure(res, bars, signals=sig)
    bars8 = [t for t in _traces(fig, "x8") if t.type == "bar" and t.x is not None and t.x[0] is not None]
    by_name = {t.name: np.asarray(t.y, float) for t in bars8}
    sides = [s for s in ("LONG", "SHORT") if any(t.side == s for t in res.trades)]
    np.testing.assert_allclose(by_name["before costs"], [sum(t.gross_pnl for t in res.trades if t.side == s)
                                                         for s in sides])
    np.testing.assert_allclose(by_name["after costs"], [sum(t.net_pnl for t in res.trades if t.side == s)
                                                        for s in sides])
    assert list(fig.layout.xaxis8.ticktext)[0].startswith("▲" if sides[0] == "LONG" else "▼")


def test_trade_analytics_says_which_input_a_panel_needs(viz_backtest):
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, _, _ = viz_backtest
    fig = trade_analytics_figure(res, bars)                     # the old call still works
    assert sorted(T.empty_panels(fig)) == ["y7", "y9"]
    notes = [a.text for a in fig.layout.annotations if "needs" in a.text]
    assert notes.count("needs the signals (signals=)") == 2
    fig = trade_analytics_figure(res)
    assert "y6" in T.empty_panels(fig) and any(a.text == "needs the bars" for a in fig.layout.annotations)


def test_trade_analytics_with_no_trades_is_one_short_message(viz_backtest):
    """Not nine empty panels with a 'no trades' note each: one short figure, one message, no axes."""
    import dataclasses

    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    empty = dataclasses.replace(res, strategy="threshold_spike", trades=[], decisions=[],
                                summary=dict(res.summary, n_trades=0))
    fig = trade_analytics_figure(empty, bars, signals=sig)
    assert len(fig.data) == 0 and "0 trades" in fig.layout.title.text
    assert fig.layout.height <= 300
    (note,) = fig.layout.annotations
    assert "placed no orders" in note.text and "nothing to analyse" in note.text
    assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False
    assert not any(k.startswith(("xaxis", "yaxis")) and k not in ("xaxis", "yaxis") for k in fig.layout.to_plotly_json())


def test_trade_analytics_predicted_vs_realised_is_like_for_like(viz_backtest):
    """With the horizon steps, the predicted h1 move is compared with the realised move over the
    SAME h1 bars from the decision close (not with the trade's own, shorter or longer, hold)."""
    from neural_trade.visualization.trade_analytics import trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    tr = res.trades
    fig = trade_analytics_figure(res, bars, signals=sig, horizon_steps=(10, 15, 20))
    close = np.asarray(sig.close, float)
    want = {}
    for s in ("LONG", "SHORT"):
        d = np.array([t.entry_bar - 1 for t in tr if t.side == s])
        d = d[d + 15 < len(close)]
        want[s.lower()] = (1 if s == "LONG" else -1) * (close[d + 15] - close[d])
    pts = {t.name: np.asarray(t.y, float) for t in _traces(fig, "x9") if t.mode == "markers"}
    for name, y in pts.items():
        np.testing.assert_allclose(y, want[name], rtol=1e-5, atol=1e-2)
    assert "same 15 bars" in fig.layout.yaxis9.title.text
    assert "h1" in fig.layout.annotations[8].text and "sign right" in fig.layout.annotations[8].text
    # without the steps the panel says it compares with the move over the trade's hold
    fig2 = trade_analytics_figure(res, bars, signals=sig)
    assert "hold" in fig2.layout.yaxis9.title.text and "hold" in fig2.layout.annotations[8].text
    held = np.concatenate([np.asarray(t.customdata, float)[:, 2] for t in _traces(fig2, "x9") if t.mode == "markers"])
    assert f"median {np.median(held):.0f} bars" in fig2.layout.yaxis9.title.text


def _templates(fig):
    for t in fig.data:
        for key in ("hovertemplate", "texttemplate"):
            v = getattr(t, key, None)
            if v is None:
                continue
            yield from ([v] if isinstance(v, str) else list(v))


def test_no_hover_spec_starts_with_a_plus_sign(viz_backtest):
    """plotly turns '%{y:+.2f}' into the invalid d3 spec '~+.2f' and prints the raw float: a signed
    spec needs an align char first ('%{y:>+.2f}')."""
    import dataclasses

    from neural_trade.visualization.trade_analytics import strategy_comparison_figure, trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    null = {"n_seeds": 5, "trade_rate": 0.01, "hold_bars": 5, "size_frac": 1.0, "random_mean_total_return": -0.05,
            "random_p05_total_return": -0.08, "random_p95_total_return": -0.02, "percentile_total_return": 40.0,
            "percentile_gross_return": 70.0}
    figs = [trade_analytics_figure(res, bars, signals=sig, horizon_steps=(10, 15, 20)),
            trade_analytics_figure(res, bars, signals=sig),
            strategy_comparison_figure({"a": dataclasses.replace(res, baselines={"random_same_freq": null}),
                                        "buy_and_hold": res}, bars)]
    specs = [s for f in figs for s in _templates(f)]
    assert specs and any(":>+" in s for s in specs)
    bad = [s for s in specs if ":+" in s]
    assert bad == []


def test_reference_lines_are_dashed_in_the_keys_too(viz_backtest):
    """Named 'dash' at width 1 draws a 9 px dash: a 30 px key swatch then looks solid. Every dashed
    line uses an explicit px pattern."""
    from neural_trade.visualization.trade_analytics import strategy_comparison_figure, trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    for fig in (trade_analytics_figure(res, bars, signals=sig),
                strategy_comparison_figure({"a": res, "buy_and_hold": res, "flat": _flat_result(res, "flat")}, bars)):
        dashes = {t.line.dash for t in fig.data if t.type == "scatter" and t.line is not None and t.line.dash}
        assert dashes and all(d == "solid" or "px" in d for d in dashes), dashes


# ------------------------------------------------------------------ several strategies
def _flat_result(res, name):
    import dataclasses

    return dataclasses.replace(res, strategy=name, trades=[], decisions=[],
                               equity=np.full_like(res.equity, res.config.initial_equity),
                               summary=dict(res.summary, n_trades=0, total_return=0.0, gross_pnl=0.0, costs_paid=0.0),
                               baselines={})


def test_strategy_comparison_layout(viz_backtest):
    import dataclasses

    from neural_trade.visualization.trade_analytics import strategy_comparison_figure

    res, bars, _, _ = viz_backtest
    worse = dataclasses.replace(res, strategy="worse", summary=dict(res.summary, total_return=res.summary["total_return"] - 0.1))
    results = {"a": res, "worse": worse, "always_flat": _flat_result(res, "always_flat"),
               "threshold_spike": _flat_result(res, "threshold_spike")}
    fig = strategy_comparison_figure(results, bars)
    assert T.empty_panels(fig) == []
    # the no-trade strategies share ONE line (drawn once, so neither hides the other) and one row
    flat = [t for t in fig.data if t.name and t.name.startswith("no trades")]
    assert len(flat) == 1 and "always_flat" in flat[0].name and "threshold_spike" in flat[0].name
    ticks = list(fig.layout.yaxis2.ticktext)
    assert len(ticks) == 3 and sum(t.startswith("no trades (2)") for t in ticks) == 1
    # rows sorted by return after costs (bottom = worst; no trades = 0%), labels on the right
    want = sorted([("worse", worse.summary["total_return"]), ("a ", res.summary["total_return"]),
                   ("no trades", 0.0)], key=lambda r: (r[1], r[0] != "no trades"))
    assert [t[:len(w)] for t, (w, _) in zip(ticks, want)] == [w for w, _ in want]
    assert fig.layout.yaxis2.side == "right"
    # the legend keys for the bars are neutral swatches, and no horizon or status colour is a strategy
    keys = {t.name: t for t in fig.data if t.type == "bar" and t.showlegend is not False}
    assert set(keys) == {"before costs", "after costs"}
    assert all(k.marker.color in (T.NEUTRAL, T.rgba(T.NEUTRAL, 0.15)) for k in keys.values())
    lines = [t for t in fig.data if t.type == "scatter" and t.mode == "lines" and (t.xaxis or "x") == "x"]
    used = {t.line.color for t in lines}
    assert not used & (set(T.HORIZON_COLORS.values()) | {T.GOOD, T.CRITICAL, T.SERIES[5]})
    assert len(fig.to_json()) < 600_000


def test_strategy_comparison_palette_and_key(viz_backtest):
    """Model strategies: pink, violet, amber (no horizon or status colour, not the red next to
    CRITICAL, not the grey next to NEUTRAL); baselines in neutral inks. One key, UNDER the plots and
    anchored to the figure's bottom, so it cannot grow into the subtitle when it wraps."""
    import dataclasses

    from neural_trade.visualization.trade_analytics import _STRATEGY_COLORS, strategy_comparison_figure

    res, bars, _, _ = viz_backtest
    names = ["calibrated_quantile", "liberal", "enhanced_multi_horizon", "fourth", "buy_and_hold", "random_signal"]
    results = {n: dataclasses.replace(res, strategy=n) for n in names}
    fig = strategy_comparison_figure(results, bars)
    eq = {t.legendgroup: t.line for t in fig.data if t.type == "scatter" and (t.xaxis or "x") == "x"}
    model = [eq[n].color for n in names[:4]]
    assert model[:3] == list(_STRATEGY_COLORS) and model[3] == _STRATEGY_COLORS[0] and eq["fourth"].dash != "solid"
    assert eq["buy_and_hold"].color == T.NEUTRAL and eq["random_signal"].color == T.INK_2
    banned = set(T.HORIZON_COLORS.values()) | {T.GOOD, T.CRITICAL, T.WARNING, T.SERIES[5], "#e66767", "#8a8984"}
    assert not {c.lower() for c in model} & {c.lower() for c in banned}
    assert _STRATEGY_COLORS[0] != T.SHORT_COLOR           # the first strategy is not the "short" amber
    # one key, at the bottom of the figure (container coordinates), growing upward into its own margin
    assert {t.legend or "legend" for t in fig.data} == {"legend"}
    lg = fig.layout.legend
    assert (lg.yref, lg.y, lg.yanchor, lg.orientation) == ("container", 0, "bottom", "h")
    # the top margin holds the title and every subtitle line (the title does not wrap)
    n_lines = 1 + fig.layout.title.text.count("<br>")
    assert fig.layout.margin.t >= 14 + 20 * n_lines + 20
    # a random_signal run without a label says it is one unmatched draw
    assert any(t.startswith("random_signal (one draw") and "not matched" in t for t in fig.layout.yaxis2.ticktext)


def test_strategy_comparison_draws_the_matched_random_null(viz_backtest):
    import dataclasses

    from neural_trade.visualization.trade_analytics import strategy_comparison_figure

    res, bars, _, _ = viz_backtest
    null = {"n_seeds": 20, "trade_rate": 0.01, "hold_bars": 5, "size_frac": 1.0, "random_mean_total_return": -0.05,
            "random_p05_total_return": -0.08, "random_p95_total_return": -0.02, "percentile_total_return": 40.0,
            "percentile_gross_return": 70.0}
    r = dataclasses.replace(res, baselines={"random_same_freq": null})
    fig = strategy_comparison_figure({"a": r}, bars)
    d = [t for t in fig.data if getattr(t.marker, "symbol", None) == "diamond-open"]
    assert len(d) == 1 and d[0].x[0] == pytest.approx(-5.0)
    assert d[0].error_x.array[0] == pytest.approx(3.0) and d[0].error_x.arrayminus[0] == pytest.approx(3.0)
    assert "beats 40% of random" in fig.layout.yaxis2.ticktext[0]
    # a null that is not size-matched is not drawn for a strategy that sizes down
    sized = dataclasses.replace(r, decisions=[dict(d, size=0.5) for d in res.decisions],
                                baselines={"random_same_freq": {k: v for k, v in null.items() if k != "size_frac"}})
    fig = strategy_comparison_figure({"a": sized}, bars)
    assert not [t for t in fig.data if getattr(t.marker, "symbol", None) == "diamond-open"]
    # explicit nulls= and labels= win
    fig = strategy_comparison_figure({"a": res}, bars, nulls={"a": null}, labels={"a": "A (knobs)"})
    assert fig.layout.yaxis2.ticktext[0].startswith("A (knobs)")
    assert [t for t in fig.data if getattr(t.marker, "symbol", None) == "diamond-open"]


def test_strategy_comparison_old_call(viz_backtest):
    from neural_trade.visualization.trading_dashboard import strategy_comparison_figure

    res, bars, _, _ = viz_backtest
    fig = strategy_comparison_figure({"a": res, "b": res}, bars)
    assert T.empty_panels(fig) == []
