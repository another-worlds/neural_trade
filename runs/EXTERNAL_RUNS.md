# External runs

Run ids that are cited in the repository but were made on another machine, so no run directory exists
here. `scripts/check_run_evidence.py` lets a listed id pass without a directory and counts it as
external; a listed id that does have a directory is checked like any other. This file is a list of
declarations, not a citing file. Every entry is reviewed by the lead (docs/RUNBOOK.md "Run directories in git").

| run id | where it ran | why it is not here | citing record |
|---|---|---|---|
| 20260927T181235Z-7785ec9-1bb2f8ee | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/core-config-cli.md |
| 20260927T204649Z-71a0fd2-2f6f3ff7 | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/overhaul-structure.md |
| 20260927T230603Z-71a0fd2-6f1d21e8 | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/static-tools-sweep.md |
| 20260927T235315Z-1a2457a-0ee3b22f | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/core-config-cli.md |
| 20260928T000341Z-1a2457a-34afd9c8 | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/core-config-cli.md |
| 20260928T035957Z-ffb1b67-90888815 | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/overhaul-structure.md |
| 20260928T040901Z-ffb1b67-6f1d21e8 | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/static-tools-sweep.md |
| 20260928T044154Z-ffb1b67-34afd9c8 | remote review session (PR #15, 2026-09-27/28), cloud machine | not reproducible here | docs/research/2026-09-28-cpu-review-sweep/findings/core-config-cli.md |
