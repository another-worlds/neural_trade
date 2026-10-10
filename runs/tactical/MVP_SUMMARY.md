# Tactical session 2026-10-06..10: what the MVP should take

For the MVP lead. Each item cites its journal entry in runs/tactical/LOG.md. Nothing here changes a default:
each candidate becomes an MVP backlog item and a paired test under D-025 first.

## Findings that should change MVP plans

1. **Direction at 10-20 min on BTC/USDT 1-minute has a ceiling near 0.57 mean-of-3 AUC, and a 13-feature logistic
   regression reaches it.** The default network scores 0.52-0.55 on the same blocks (H23, H27: +0.036 [+0.006,+0.067]
   for the regression). Rebuilt networks (MLP, GRU, PatchTST-lite, TCN), 3 years of training and seed ensembles add at
   most +0.003 (H28, H35, H36, H44). Candidate items: a regression direction model in the registry; the leaderboard's
   baseline row against it (NT-033's frozen twin is not the strongest baseline).
2. **Learnable indicator periods add nothing in the window model.** No indicators = 14 learned families on direction;
   learned = textbook periods (H21, H22, H25, H26, H37). This is evidence against D-031's premise in this setup and is
   a question for the owner before NT-050 spends GPU on indicator-period sweeps.
3. **The audit of the default network** (runs/tactical/rebuild/AUDIT.md): unnormalised indicator channels (std
   0.015-358), a LayerNorm over time that removes levels, a tower shared by all heads, epoch selection on the composite
   loss, dead computation every step. H42 corrects one audit point: the direction loss carries 30-45% of the trunk
   gradient, not ~10% (H16 was one run, 8 batches).
4. **Networks earn their keep on volatility, not direction.** The variance tower and a JEPA embedding rank future
   |move| better than HAR-RV with time of day (+0.016..+0.019 Spearman on minutes, +0.045 hourly boosting), but not on
   QLIKE (H39, H41). Candidate: a volatility model judged against HAR-RV, not against trailing RV60.

## Process lessons

- **Held-out checks kill most wins:** the hill-climb's 59.6% tail fell to 54.3% (H30); the hourly triple barrier went
  from +32 bps on dev to +9.8 on the held-out period and failed (H38); no regime filter survived (H40).
- **A GPU driver reset kills every GPU process:** three heavy (persistent-tape) processes on the 12 GB card coincided
  with two resets; one alone runs clean (H43). At most 2 heavy processes at once.
- **Speed:** parallel processes on one GPU do not raise throughput (time-slicing); cut epoch cost instead (H10, night log).
- **Tools to reuse:** the lab (runs/tactical/lab/: seconds per experiment on the network's exact blocks), the honest
  split (thresholds on the first half of a val block, test on the second), the graph-mode gradient probe (2x faster, H42).
