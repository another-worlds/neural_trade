"""RunContext: one directory per training run, so every reported number links to a run.

    ctx = RunContext.create(config, tags=["ablation", "all_on"])
    result = train_and_evaluate(config=ctx.config, run_context=ctx, ...)

creates ``runs/<UTC-timestamp>-<git sha>[-dirty]-<config hash>[-name]/`` holding

    config.yaml            the exact configuration (flat keys)
    env.json               versions, CUDA build, devices, git state
    meta.json              seed, tags, run id
    metrics.jsonl          one line per epoch (telemetry.JsonlEpochLogger)
    status.json            progress, seconds per step, telemetry errors
    training_log.csv       Keras CSVLogger
    weights.h5 / scaler.joblib
    artifacts/             the serving bundle (training.artifacts.ArtifactBundle)
    eval_report_*.json/md  evaluation reports (evaluation.report)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from neural_trade.core.config import Config
from neural_trade.utils.env import fingerprint, git_sha


def config_hash(config: Config) -> str:
    return hashlib.sha256(config.to_yaml().encode("utf-8")).hexdigest()[:8]


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    config: Config
    seed: int
    tags: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, config: Config, *, root="runs", seed: Optional[int] = None, tags=(), name: Optional[str] = None,
               write_env: bool = True) -> "RunContext":
        cfg = config.copy()
        seed = int(cfg.SEED if seed is None else seed)
        cfg.override(SEED=seed)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{git_sha()}-{config_hash(cfg)}" + (f"-{name}" if name else "")
        run_dir = Path(root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg.override(MODEL_PATH=str(run_dir / "weights.h5"), SCALER_PATH=str(run_dir / "scaler.joblib"),
                     ARTIFACTS_DIR=str(run_dir / "artifacts"))
        ctx = cls(run_id, run_dir, cfg, seed, list(tags))
        cfg.to_yaml(run_dir / "config.yaml")
        (run_dir / "meta.json").write_text(json.dumps(
            {"run_id": run_id, "seed": seed, "tags": ctx.tags, "created_utc": stamp}, indent=2), encoding="utf-8")
        if write_env:
            (run_dir / "env.json").write_text(json.dumps(fingerprint(), indent=2, default=str), encoding="utf-8")
        return ctx

    @classmethod
    def load(cls, run_dir) -> "RunContext":
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(meta["run_id"], run_dir, Config.from_yaml(run_dir / "config.yaml"), meta["seed"], meta.get("tags", []))

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def write_json(self, name: str, obj) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
        return p
