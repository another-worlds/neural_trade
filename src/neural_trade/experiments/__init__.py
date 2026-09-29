"""Experiments: run tracking, comparison, the experiment engine and the frozen ablation harness.

The experiment engine (NT-026): ``scenario`` (the YAML scenario and sweep spec), ``runner`` (the
resumable runner, ``neural-trade scenario run``), ``store`` (the run store and its sqlite index),
``scorer`` (one scorer: the eval report of each fold's out-of-sample block plus the backtest) and
``dataset`` (the dataset fingerprint and fold layout). ``ablation`` is frozen as history (D-023).
"""
