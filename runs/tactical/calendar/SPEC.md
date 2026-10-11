# Calendar effects in BTC/USDT hourly returns: SPEC (pre-registered, committed before any result)

Tactical session, exploratory, CPU only. Written 2026-10-11 before any bucket statistic was computed.

## Data and return
- `Bitcoin_BTCUSDT.csv` (1-minute, UTC). Hour t = the bar stamped hh:00. Return r_t = open(t+1h) / open(t) - 1 in bps:
  entry at an hour's open, exit at the next hour's open (next-open to next-open, 1-hour hold). A row is dropped when either
  hh:00 bar is missing. Trades in consecutive hours never overlap by construction.

## Hypotheses and buckets (36 in total; Holm over all 36)
(a) hour-of-day: 24 buckets, hour h = return of the hour starting h:00 UTC.
(b) day-of-week: 7 buckets (UTC weekday of the hour's start).
(c) funding (Binance perpetual 00:00/08:00/16:00 UTC): "pre" = hours starting 07:00, 15:00, 23:00; "post" = hours starting 00:00, 08:00, 16:00. 2 buckets. (post - pre) is reported descriptively, not tested.
(d) month edges: last UTC day of the month; first UTC day of the month. 2 buckets.
(e) weekend: Saturday + Sunday UTC. 1 bucket.
Buckets overlap (an hour is in several); that is accepted, Holm is applied to all 36 regardless.

## Selection period: 2021-01-01 .. 2024-06-30 (hours starting inside)
- Effect = mean r over the bucket's hours, in bps. 95% interval and p-value by Newey-West (Bartlett, 48 hourly lags) on the mean of the
  bucket's hours; normal approximation. Holm-Bonferroni at family level 0.05 over the 36 p-values (two-sided).
- Calendar strategy (frozen from the selection period only): the set S of buckets significant after Holm, each with the sign of its
  selection mean. For every hour the signal is the sum of the signs of the buckets in S that contain it; positive = long, negative = short,
  zero or no bucket = no trade. If S is empty there is no strategy, and that is the result.

## One look: 2024-07-01 .. 2025-09-29 (the data end)
- Trade every hour with a signal. Report for round-trip costs 0, 1 and 2 bps: trades, net bps per trade, hit rate, monthly net-bps sum with a
  t 95% interval over months, max drawdown (equity compounding the per-trade net return, 1 unit of capital fully invested per trade).
- Nothing is changed after this look. If S is empty, the test period shows only the descriptive per-bucket means (labelled descriptive).
- Caveat: 2024-07..2025-09 was already used for other hypotheses (H38, H56). This is a fresh look for a distinct hypothesis only; it is not an untouched block.

## Descriptive (no decision)
- Each bucket's mean per calendar year 2017..2025 with its NW 95% interval, and the sign agreement across years.

## Reading rule fixed in advance
- "Effect exists" needs Holm significance on selection AND a test-period net bps per trade at 1 bps cost above 0 with its monthly interval excluding 0. Anything else is "no tradable calendar effect".
- A negative result is recorded as it stands.

## Tests (seconds)
- returns are next-open to next-open; no selection step reads data after 2024-06-30.
