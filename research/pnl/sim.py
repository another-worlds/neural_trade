"""Simulators (SPEC 'simulator'). Engine E is the ONLY path to reported PnL; engine F is the flat-cost training proxy.

ENGINE E (event-driven, 1-min bars, compounding equity, signed BTC contracts q, exposure x = q c / E)
- A decision at the close of bar t gives a target x* in [-1, 1]. dq = (x* - x) E / c_t is split into a REDUCE leg (toward
  0, incl. the closing leg of a flip) and an INCREASE leg.
- Order types by fee scenario:
    C7 (primary): REDUCE taker at the next open; INCREASE maker at L = c_t.
    C4: both legs maker; an unfilled exit (REDUCE) chases to taker after K bars.
    C10: both legs taker at the next open.
- Maker fill: first bar j in [t+1, t+K] with low_j <= L - tick (buy) or high_j >= L + tick (sell); price L, all-or-none.
  Unfilled INCREASE: cancelled when the policy says so (Track 1 entries), otherwise chased to taker at o_{t+K+1}.
- Fill scenarios: B (base; tick 0.1 USDT; no slippage); S (stress: trade-through 1 bps instead of 1 tick, +1 bar latency,
  +1 bps on every taker fill); N (nominal: every fill at the decision close c_t at its nominal fee; reconciliation only).
- Fees: maker 2 bps, taker 5 bps of filled notional. Funding: at a bar carrying rate f, E -= f * q * c_{prev}
  (the position held at the end of the previous bar pays; longs pay positive rates).
- 1x cap: target clipped to [-1, 1]; if |x| > 1.10 at a bar close, trim to 1.0 as taker at the next open (logged).
- A new decision cancels any working order. Positions are closed (taker) at the end of the run.
- Gaps: no decisions for 60 bars after a gap; positions carry and are marked at the first bar after it.
- Scale: equity, position and open orders are renormalised to E = 1 at every UTC day start (all PnL terms are linear
  in size, so this is exact). Daily log returns stay precise even for ruinous rules; fees and funding totals are sums
  of daily amounts in units of start-of-day equity (fractions of equity).

ENGINE F (vectorised): net_k = x_k (R_k - F_k) - (c_in inc_k + c_out red_k) on a decision grid, exact inc/red split.

Policy protocol (duck-typed):
    policy.K              maker window in bars (policy.K_at(i), if defined, overrides it per decision)
    policy.chase_entries  True: unfilled INCREASE orders chase to taker after K; False: they are cancelled
    policy.decide(i, x, st) -> target exposure or None (no action), called at every bar close i in [start, end)
    policy.on_fill(i, kind, st) (optional) notified of fills ('increase'/'reduce') and cancellations ('cancel')
`st` is the SimState below.
"""
import numpy as np

MAKER, TAKER = 2e-4, 5e-4
TICK = 0.1
DAY = 86400


class SimState:
    __slots__ = ("q", "E", "x", "working", "entry_fill_bar", "entry_side", "pending_target", "last_cancel_bar")

    def __init__(self):
        self.q = 0.0
        self.E = 1.0
        self.x = 0.0
        self.working = None          # dict(side, qty, limit, start, end, kind, chase)
        self.entry_fill_bar = -1
        self.entry_side = 0
        self.pending_target = None
        self.last_cancel_bar = -1


def simulate(b, policy, start, end, fee="C7", fill="B", close_at_end=True, record_exposure=False):
    """Run engine E on bars b over bar indices [start, end). Returns a result dict (see bottom)."""
    o, h, l, c, fund, t = b.o, b.h, b.l, b.c, b.funding, b.t
    gap = b.gap_before
    lat = 1 if fill == "S" else 0
    slip = 1e-4 if fill == "S" else 0.0
    K = int(policy.K)
    chase_entries = bool(policy.chase_entries)
    st = SimState()
    mark = c[start - 1] if start > 0 else o[start]
    E = 1.0
    q = 0.0
    fees_total = funding_total = turnover = 0.0
    trims = 0
    n_taker = n_maker = n_cancel = n_chase = 0
    taker_queue = []                 # list of (exec_bar, dq, kind)
    trades, open_trade = [], None
    fills = []                       # (bar, kind, dq, price, fee_rate)
    day_log, day_ids = [], []
    day_fees, day_fund, day_acc = [], [], [0.0, 0.0]   # per day, in units of start-of-day equity
    expo = np.zeros(end - start, np.float32) if record_exposure else None
    suppress_until = -1
    trim_pending = -1
    cur_day = t[start] // DAY

    def fill_trade(px, dq, fee_rate, i, kind):
        nonlocal E, q, mark, fees_total, turnover, open_trade
        E += q * (px - mark)
        mark = px
        fee_amt = fee_rate * abs(dq) * px
        E -= fee_amt
        fees_total += fee_amt
        day_acc[0] += fee_amt
        fills.append((i, kind, dq, px, fee_rate))
        notional_x = abs(dq) * px / max(E, 1e-12)
        turnover += notional_x
        q_old = q
        q = q + dq
        if abs(q) * px < 1e-12 * max(E, 1e-300):
            q = 0.0
        # trade bookkeeping (round trips: from flat to flat, per side)
        if q_old == 0.0 and q != 0.0:
            open_trade = {"entry_bar": i, "side": int(np.sign(q)), "entry_px": px, "fees": fee_amt, "funding": 0.0,
                          "E0": E + fee_amt}
        elif open_trade is not None:
            open_trade["fees"] += fee_amt
            if q == 0.0 or np.sign(q) != open_trade["side"]:
                tr = open_trade
                tr["exit_bar"], tr["exit_px"] = i, px
                gross = tr["side"] * np.log(px / tr["entry_px"]) * 1e4
                fee_bps = tr["fees"] / tr["E0"] * 1e4
                fund_bps = tr["funding"] / tr["E0"] * 1e4
                trades.append({"entry_bar": tr["entry_bar"], "exit_bar": i, "side": tr["side"], "gross_bps": gross,
                               "fees_bps": fee_bps, "funding_bps": fund_bps, "net_bps": gross - fee_bps - fund_bps,
                               "hold_minutes": int(i - tr["entry_bar"])})
                open_trade = None
                if q != 0.0:          # flip: the new leg opens a new trade
                    open_trade = {"entry_bar": i, "side": int(np.sign(q)), "entry_px": px, "fees": 0.0,
                                  "funding": 0.0, "E0": E}
        st.q = q
        if q != 0.0 and np.sign(q) != np.sign(q_old):
            st.entry_fill_bar, st.entry_side = i, int(np.sign(q))
        if hasattr(policy, "on_fill"):
            policy.on_fill(i, kind, st)

    for i in range(start, end):
        # new UTC day bookkeeping (equity at the end of the previous day)
        d = t[i] // DAY
        if d != cur_day:
            E_end = E + q * (c[i - 1] - mark)
            day_log.append(np.log(max(E_end, 1e-300)))
            day_ids.append(cur_day)
            day_fees.append(day_acc[0])
            day_fund.append(day_acc[1])
            day_acc[0] = day_acc[1] = 0.0
            cur_day = d
            # renormalise the scale to E = 1 (exact: every PnL term is linear in size)
            f = 1.0 / max(E_end, 1e-300)
            E, q = E_end * f, q * f
            mark = c[i - 1]
            st.q, st.E = q, E
            taker_queue = [(eb, dq_ * f, kd) for (eb, dq_, kd) in taker_queue]
            if st.working is not None:
                st.working["qty"] *= f
            if open_trade is not None:
                for key in ("fees", "funding", "E0"):
                    open_trade[key] *= f
        # 1. funding at the open of bar i, paid by the position held at the end of bar i-1
        if fund[i] != 0.0 and q != 0.0:
            amt = fund[i] * q * c[i - 1]
            E -= amt
            funding_total += amt
            day_acc[1] += amt
            if open_trade is not None:
                open_trade["funding"] += amt
        if gap[i]:
            suppress_until = i + 60
        # 2. taker orders scheduled for this bar's open
        if taker_queue:
            rest = []
            for (eb, dq, kind) in taker_queue:
                if eb == i:
                    px = o[i] * (1 + slip * np.sign(dq))
                    fill_trade(px, dq, TAKER, i, kind)
                    n_taker += 1
                else:
                    rest.append((eb, dq, kind))
            taker_queue = rest
        # 3. working maker order: trade-through check in bar i
        w = st.working
        if w is not None and i >= w["start"]:
            thr = w["limit"] * (1e-4 if fill == "S" else 0.0) + (0.0 if fill == "S" else TICK)
            hit = (l[i] <= w["limit"] - thr) if w["qty"] > 0 else (h[i] >= w["limit"] + thr)
            if hit:
                fill_trade(w["limit"], w["qty"], MAKER, i, w["kind"])
                n_maker += 1
                st.working = None
            elif i >= w["end"]:
                if w["chase"]:
                    taker_queue.append((i + 1, w["qty"], w["kind"]))
                    n_chase += 1
                else:
                    n_cancel += 1
                    st.last_cancel_bar = i
                    if hasattr(policy, "on_fill"):
                        policy.on_fill(i, "cancel", st)
                st.working = None
        # 4. mark to the close
        E += q * (c[i] - mark)
        mark = c[i]
        x = q * c[i] / E if E > 0 else 0.0
        st.E, st.x = E, x
        if record_exposure:
            expo[i - start] = x
        # 5. drift trim
        if abs(x) > 1.10 and i >= trim_pending:
            target_q = np.sign(q) * 1.0 * E / c[i]
            taker_queue.append((i + 1 + lat, target_q - q, "reduce"))
            trim_pending = i + 1 + lat + 1
            trims += 1
        # 6. decision at the close of bar i; the policy sees the exposure including committed taker orders
        if i < end - 1 and i >= suppress_until:
            q_eff = q + sum(dq_ for (_, dq_, _) in taker_queue)
            x_eff = q_eff * c[i] / E if E > 0 else 0.0
            tgt = policy.decide(i, x_eff, st)
            if tgt is not None:
                tgt = float(np.clip(tgt, -1.0, 1.0))
                st.working = None                          # a new decision cancels the working order
                K = int(policy.K_at(i)) if hasattr(policy, "K_at") else K
                dq = (tgt - x_eff) * E / c[i]
                if abs(dq) * c[i] / E > 1e-6:
                    red = 0.0
                    if q_eff != 0.0 and np.sign(dq) != np.sign(q_eff):
                        red = np.sign(dq) * min(abs(dq), abs(q_eff))
                    inc = dq - red
                    tol = 1e-9 * E / c[i]
                    if abs(inc) < tol:
                        inc = 0.0
                    if abs(red) < tol:
                        red = 0.0
                    dq = red + inc
                    kind = "reduce" if red != 0.0 and inc == 0.0 else ("flip" if red != 0.0 else "increase")
                    if red == 0.0 and inc == 0.0:
                        pass                                   # float residue only: no order
                    elif fill == "N":
                        if red != 0.0:
                            fill_trade(c[i], red, MAKER if fee == "C4" else TAKER, i, "reduce")
                        if inc != 0.0:
                            fill_trade(c[i], inc, TAKER if fee == "C10" else MAKER, i, "increase")
                    elif fee == "C10":
                        taker_queue.append((i + 1 + lat, dq, kind))
                    elif fee == "C4":
                        st.working = {"qty": dq, "limit": c[i], "start": i + 1 + lat, "end": i + K + lat,
                                      "kind": kind, "chase": True if red != 0.0 else chase_entries}
                    else:                                  # C7: reduce taker, increase maker
                        if red != 0.0:
                            taker_queue.append((i + 1 + lat, red, "reduce"))
                        if inc != 0.0:
                            st.working = {"qty": inc, "limit": c[i], "start": i + 1 + lat, "end": i + K + lat,
                                          "kind": "increase", "chase": chase_entries}
    # close at the end of the run (taker at the last close)
    if close_at_end and q != 0.0:
        fill_trade(c[end - 1], -q, TAKER, end - 1, "reduce")
    day_log.append(np.log(max(E + q * (c[end - 1] - mark), 1e-300)))
    day_ids.append(cur_day)
    day_fees.append(day_acc[0])
    day_fund.append(day_acc[1])
    daily = np.array(day_log)                      # each day starts at E = 1 (renormalised), so log(E_end) is its return
    eq = np.exp(np.cumsum(daily))
    return {"daily": daily, "days": np.array(day_ids), "equity": eq, "trades": trades, "fees": fees_total,
            "funding": funding_total, "turnover": turnover, "trims": trims, "n_taker": n_taker, "n_maker": n_maker,
            "n_cancel": n_cancel, "n_chase": n_chase, "exposure": expo, "fills": fills, "final_E": float(eq[-1]),
            "daily_fees": np.array(day_fees), "daily_funding": np.array(day_fund),
            "start": int(start), "end": int(end), "fee": fee, "fill": fill}


def engine_f(x, R, F, c_in=3e-4, c_out=5e-4, x0=0.0):
    """Flat-cost net per decision step. x: exposures after decision k (held over step k); R: simple return over the
    step; F: funding rate summed over the step. inc/red split exact incl. flips."""
    x = np.asarray(x, dtype=np.float64)
    xp = np.r_[x0, x[:-1]]
    dx = np.abs(x - xp)
    inc = (np.abs(x) + dx - np.abs(xp)) / 2
    red = (np.abs(xp) + dx - np.abs(x)) / 2
    return x * (R - F) - (c_in * inc + c_out * red)


class ConstPolicy:
    """Hold a constant target from the first decision (B&H, cash, static-exposure baselines)."""

    def __init__(self, x, K=15, chase=True):
        self.x, self.K, self.chase_entries, self.done = x, K, chase, False

    def decide(self, i, x, st):
        if not self.done:
            self.done = True
            return self.x
        return None


class TargetPolicy:
    """Precomputed target per bar (np.nan = no action); used for tests and simple rules."""

    def __init__(self, targets, start, K=1, chase=False):
        self.targets, self.start, self.K, self.chase_entries = targets, start, K, chase

    def decide(self, i, x, st):
        v = self.targets[i - self.start]
        return None if np.isnan(v) else float(v)
