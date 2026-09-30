# Micro loop (D-041): report

Owner goals: raise predictive power and strategy PnL through a fast micro-scale loop (D-041, 2026-09-29), with
the target "stable predictive power above 60% and drawdown below 5%" (/goal, 2026-09-29). The journal with
every hypothesis, its method, cost, evidence path and result: [LOG.md](LOG.md).

## Verdict

**The target is not reached, and on the information available to this project it is not reachable.** Twenty-one
lines of attack were tested with committed evidence. Directional predictability of BTC/USDT from 1-minute price,
volume, spot order flow and futures signals peaks at AUC about 0.52-0.54 (51-53% of calls right; 54-56% on the
most confident tenth), at horizons from 10 minutes to 5 hours. The neural network never beats a logistic
regression on three lagged returns, and at 1 hour it is significantly below it (z -3.3 to -3.5 in every level-2
variant).

## What was tested (all in LOG.md)

| Area | Lines of attack | Best number |
|---|---|---|
| Trading the existing signal | selectivity, longer holds, confidence buckets, 20-strategy study (NT-005) | gross edge ~1 bps per trade vs 26 bps cost |
| What to predict | horizons 10 min-5 h, windows 60 / 240 bars, 10 vs 360 days of training | direction AUC never above a 3-lag logistic |
| Inputs | OHLCV + 14 learnable indicator families (NT-047) | no gain over close-only |
| Labels and objective | cost-sensitive labels (E1), net-P&L objective (E2, NT-087), triple-barrier labels (E3 bar) | all below or at the logistic baseline |
| Maths and hyperparameters | 960-trial screen (NT-088/NT-092), level 2 on 5 configurations | maths stable; no configuration recovers direction |
| Conditions | hour of day, volatility and volume regimes, after large moves | post-shock reversal: hit 54.2%, z 4.1 out of sample, but gross edge -0.4 bps |
| New information (evidence for owner question 8) | spot taker-buy order flow; futures basis, futures flow, futures-spot lead; futures order-book depth (+-1-5%, minute snapshots) | spot flow +0.010 AUC (z 3.2) at 15 min; futures and depth add nothing; best hit on all bars 53% (1 min, a 2 bps move) |

## What the numbers say about the target

- A stable 60% of directional calls from this information would need AUC of roughly 0.64 or more; the best
  observed is 0.54, and the published literature for intraday BTC from price, volume, blockchain and sentiment
  data is 51-56% (docs/research/2026-09-29-pnl-target/ section 2.2).
- 70%+ hit rates are published only for order-book microstructure at sub-second horizons, where a move is far
  smaller than the 13 bps per side this project pays.
- Drawdown below 5% is reachable by sizing whenever the edge is positive; with no positive net edge, only
  "do not trade" meets it.

## What could still move it (owner decisions)

1. **Tick-level order-book data** (the full book and its updates, at sub-second horizons): the one information
   source the literature ties to high hit rates. The free minute-level depth snapshots add nothing (Q8-probe-3);
   tick depth (bookTicker is ~340 MB per day) or a live collector is an owner decision, and at those horizons a move
   is far below the 13 bps per side this project pays.
2. **Lower costs**: the timing signal the models do carry is worth ~0.9-1.2 bps per trade; the zero-cost rescore
   of the 360-day run made +17% in 32 days. A venue or fee tier near zero changes the economics, not the
   predictive power.
3. **A different reading of the target**: e.g. 60% of *taken* trades on rare, large-move setups over many months
   (the P&L note's section 2.4: 300-380 trades are needed to tell 60% from 53%).
