# Tactical hyp1: direction-head switches (CPU only, no GPU run)

Branch nt-tactical-hyp1 (from nt-tactical 4953364). All defaults = today's graph; tests in
tests/test_direction_hyp1.py (log: runs/tactical/hyp1_tests.log). Applies to gru_attention and
gru_small (both use `_direction_head`; linear_indicators uses head_kit, which calls the same function).
Files: src/neural_trade/models/gru_attention.py (`_direction_head`), src/neural_trade/core/config.py, the test.
Golden run not executed (CPU-only, heads unchanged when off; test pins off == default graph, weight shapes,
outputs and loss count). Per-step path: off adds no layers/ops.

| switch | default | meaning |
|---|---|---|
| DIRECTION_HEAD_MODE | mixed | `skip_only`: deep logit is a frozen zero Dense; logit = linear skip over lag features + bias (joint logreg) |
| DIRECTION_DEEP_SHRINK | 0.0 | L2 activity penalty on the deep direction logit output (scaled by LAMBDA_INTER=1); try 0.1, 1, 10 |
| DIRECTION_DEEP_DROPOUT | 0.0 | dropout on the deep direction logit's input only (training only) |

Third idea chosen: dropout on the deep logit input. Why: variance reduction in-graph, no extra forward
passes, no callback or served-epoch logic (weight averaging/SWA would touch D-011 and the trainer; averaging
over k anchor bars needs multi-window inference and Predictor changes). Not run; no evidence of effect.

Screen spec `overrides`:
```yaml
overrides: {DIRECTION_HEAD_MODE: skip_only}
overrides: {DIRECTION_DEEP_SHRINK: 1.0}
overrides: {DIRECTION_DEEP_DROPOUT: 0.5}
```
Caveat: shrink and skip_only are the same family; skip_only is the limit of shrink -> inf.
