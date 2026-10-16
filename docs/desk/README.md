# Trading desk: how an implementer session works here

This folder is the whole brief for building the trading desk (`src/neural_trade/desk/`) with Sonnet 5.5
medium implementers under the project's work loop. Read this page, then only what your item points to.

## What you are building

A TradingView-like trading desk on top of neural_trade: a chart front end, manual trading on Binance spot
(paper by default, live behind arming), bots, an event-driven backtester that shares the bots' strategy code,
a plugin contract for classic indicators and strategies, and a Model Lab over the existing experiment engine.
The reasoning and the research behind every choice are in [PLAN.md](PLAN.md); you do not need to re-derive
them. The owner's six requirements are PLAN "Requirements" (R1-R6).

## The three documents and how to use them

| document | use it for |
|---|---|
| [TASKS.md](TASKS.md) | your item's row: title, size, dependencies, files you may change, acceptance criteria; the lane order; the per-milestone verification; the task key to NT id table |
| [CONTRACTS.md](CONTRACTS.md) | the signatures, dataclass fields, status edges, fill rules (A5), plugin contract (A7, C3), database schema (C4) and API (C5) you implement **verbatim**; if your item needs something this file does not define, stop and report (never invent an interface) |
| [PLAN.md](PLAN.md) | the architecture (A1-A11), the reuse map (which neural_trade functions to call instead of re-implementing), the research digest; read the sections your item names |

Your backlog item (`docs/BACKLOG.md`, NT-215 onward) is the authority for acceptance; TASKS.md and
CONTRACTS.md are the detail it points to.

## Rules that apply to every desk item (besides CLAUDE.md and OPERATING_MODEL)

1. **Reuse, never re-implement.** Statistics (`strategy/performance.py`, `metrics/statistics.py`),
   preprocessors (`data/preprocessors.py`), `serving.Predictor`, the experiment engine (`experiments/`), the
   figures (`visualization/`) and the theme are called, not copied. PLAN "What the research established" lists the
   exact names.
2. **Nothing in neural_trade imports `desk`** (a layering test enforces it; `cli.py` may import `desk.cli` only
   inside a function). Desk plugins live in `plugins_desk/` at the repo root, never in `plugins/`.
3. **The server never uses the GPU**; GPU work is a `neural-trade` subprocess behind the desk's own gate
   (PLAN A1.4, A3). Nothing you add runs inside a training step (D-018). The golden record must not move.
4. **Fill rules are a contract** (CONTRACTS A5, pinned by the parity test DESK-1.9). Changing one is an owner
   decision, not a fix.
5. **Honest numbers**: `direction_signal == 'none'` is shown as "no usable direction signal", served delta
   n/a when beta = 0, every report carries buy-and-hold, always-flat and the matched random null with n_eff
   bands; test-fold columns never rank or sort.
6. **Desk state lives in `var/desk/`** (gitignored), never under `runs/` or `data/`. Secrets only from
   environment variables.
7. **Standing checks before you report:** `tests/desk/` green with `-n 8`, ruff clean, `import neural_trade.desk`
   loads no TensorFlow, fastapi, nicegui or ccxt, the fast suite unchanged, CI `desk` and `unit` green on your
   branch push. UI items cite `desk/ui/conventions.md` (created by DESK-4.1) and the two JS elements carry
   Playwright evidence from the CI `desk` job.
8. **Size:** S under 150 lines including tests, M 150-400, L 400-800. If your item is larger, stop and report;
   the lead splits it.
9. **Environment:** never install anything into `nt`. The desk extra is installed by the lead (DESK-0.3). Until
   then items that need fastapi, nicegui or ccxt wait; DESK-0.2, 0.4, 0.5, 0.6, 0.7 and 1.x do not need them.

## Order of work

Lanes (one implementer each, disjoint files; TASKS.md "Order of execution"):

- **Lane A** (engine): 0.2 -> 0.4 -> 0.5 -> 1.1 -> 1.2 -> 1.3 -> 1.4 -> 1.6 -> 1.7 -> 1.8 -> 1.9 -> 1.10 -> 1.11
- **Lane B** (data and paper trading): 0.6 -> 0.7 -> 0.8 -> 1.5 -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 2.6 -> 2.7
- **Lane C** (plugins): 7.1 (after A's 1.6) -> 7.2 -> 7.3 -> 7.4 (after A's 1.11) -> 7.5
- Then DESK-3 (API) and DESK-5 (Lab backend), then DESK-4 (front end, after the chart spike 4.0), then DESK-6.

Gates the lead owns: the owner's answers to STATUS questions 0 (stack) and 4 (environment) before DESK-0.3;
the chart spike (4.0) before 4.2.

## Where to look when stuck

- The oracle for fills and costs: `src/neural_trade/strategy/backtest.py` (`run_backtest`), with CONTRACTS A5
  as its written form.
- The subprocess pattern for jobs: `src/neural_trade/notebook/longrun.py` (`launch`, `launch_env`, `progress`,
  `pid_alive`, `tail`, `read_jsonl`).
- The headless panel logic to call from the Lab: `src/neural_trade/notebook/panel_data.py`, `panel_compare.py`,
  `control_panel.py`.
- Test fixtures: `tests/conftest.py` (`synthetic_bars`, `synthetic_close`, `real_slice`, `viz_frame`,
  `viz_backtest`, `tiny_config`).
- Owner decisions you must not cross: `docs/DECISIONS.md` D-001, D-002, D-007, D-012, D-018, D-020, D-029,
  D-044, D-050, D-063, D-065, D-066, D-067.
