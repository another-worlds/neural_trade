"""DESK-5.6: the model registry (D-067, PLAN A8 "Models", CONTRACTS C4 ``models``, C5 ``/lab/models``).

``promote`` takes a run directory (``runs/.../<id>``) or its ``artifacts`` bundle directory and records it in the desk
``models`` table with its config facts (symbol, tf = ``RESAMPLE_MINUTES``, ``LOOKBACK``, horizons in bars, the dataset
sha256 of the run's ``meta.json``) and its dev scores from the run index. A bundle is refused unless its
``meta.json`` says ``format_version == 1`` and ``config.yaml`` and ``weights.h5`` exist.

Reference or copy: the default REFERENCES the bundle in place (``bundle_dir`` = the artifacts directory, ``run_dir`` the
run), because weights are heavy and runs are never moved (OPERATING_MODEL); ``copy_to_home`` copies the bundle into
``<home>/models/<id>`` so a promoted model survives the run being pruned (``bundle_dir`` then differs from
``run_dir``, which is why C4 has both columns).

Dev scores (D-020): a run whose fold role is ``test`` is promoted with ``split: "test"``, ``rank_eligible: false`` and
NO numbers; only a dev run carries ``sharpe_net`` and the other key numbers (and the raw index scores).

``PredictorCache`` holds one ``serving.Predictor`` per bundle, loaded lazily and on CPU: it refuses to load one while
TensorFlow sees a GPU unless ``allow_gpu=True`` (the server entry point sets ``CUDA_VISIBLE_DEVICES=-1`` itself; this
module sets nothing). TensorFlow and the Predictor are imported inside functions only.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from neural_trade.desk.config import DESK_HOME, PROJECT_ROOT
from neural_trade.desk.core.model import DeskError
from neural_trade.desk.store.db import DeskDb, RowRepo

__all__ = ["BUNDLE_FORMAT", "ModelError", "GpuVisible", "ModelRegistry", "PredictorCache", "bundle_facts"]

#: ``training.artifacts.FORMAT_VERSION``; repeated here because importing that module loads TensorFlow.
BUNDLE_FORMAT = 1


class ModelError(DeskError):
    """A bundle or request the registry refuses."""


class GpuVisible(DeskError):
    """TensorFlow sees a GPU and the caller did not allow it (the desk server never uses the GPU)."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelError(f"cannot read {path.as_posix()}: {exc}") from exc
    except ValueError as exc:
        raise ModelError(f"{path.as_posix()} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ModelError(f"{path.as_posix()} is not a JSON object")
    return doc


def _locate(path: Union[str, Path]) -> tuple:
    """``(run_dir, bundle_dir)`` from a run directory (it holds ``artifacts/``) or the ``artifacts`` directory. A run
    directory has its own ``meta.json`` (the engine's), so only ``artifacts/meta.json`` counts."""
    p = Path(path).resolve()
    bundle = p if p.name == "artifacts" else p / "artifacts"
    if not (bundle / "meta.json").exists():
        raise ModelError(f"no artifacts/meta.json under {p.as_posix()}: not a trained bundle")
    return bundle.parent, bundle


def bundle_facts(bundle: Path) -> Dict[str, Any]:
    """Validate a bundle directory and return its config facts; raises ``ModelError`` for a bundle that is refused."""
    meta = _read_json(bundle / "meta.json")
    version = meta.get("format_version")
    if isinstance(version, bool) or version != BUNDLE_FORMAT:
        raise ModelError(f"{bundle.as_posix()}: meta.json format_version is {version!r}, the desk reads "
                         f"format_version == {BUNDLE_FORMAT}")
    for name in ("config.yaml", "weights.h5"):
        if not (bundle / name).exists():
            raise ModelError(f"{bundle.as_posix()}: {name} is missing")
    from neural_trade.core.config import Config

    try:
        cfg = Config.from_yaml(bundle / "config.yaml")
    except Exception as exc:        # a Config that does not load is a refused bundle, whatever the parser raised
        raise ModelError(f"{bundle.as_posix()}: config.yaml does not load: {type(exc).__name__}: {exc}") from exc
    return {"symbol": str(cfg.SYMBOL), "tf": int(cfg.RESAMPLE_MINUTES), "lookback": int(cfg.LOOKBACK),
            "horizons": [int(h) for h in cfg.HORIZON_STEPS]}


class PredictorCache:
    """One ``Predictor`` per bundle directory, loaded on first use. ``loader`` and ``gpu_visible`` are injectable."""

    def __init__(self, *, loader: Optional[Callable[[Path], Any]] = None,
                 gpu_visible: Optional[Callable[[], bool]] = None) -> None:
        self._loader = loader or self._default_loader
        self._gpu_visible = gpu_visible or self._default_gpu_visible
        self._items: Dict[tuple, Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _default_loader(bundle: Path) -> Any:
        from neural_trade.serving.predictor import Predictor

        return Predictor.from_artifacts(bundle)

    @staticmethod
    def _default_gpu_visible() -> bool:
        import tensorflow as tf

        return bool(tf.config.list_physical_devices("GPU"))

    @staticmethod
    def _key(bundle: Path) -> tuple:
        try:
            st = (bundle / "weights.h5").stat()
            return (bundle.resolve().as_posix(), st.st_size, st.st_mtime_ns)
        except OSError:
            return (bundle.resolve().as_posix(), None, None)

    def get(self, bundle_dir: Union[str, Path], *, allow_gpu: bool = False) -> Any:
        bundle = Path(bundle_dir)
        key = self._key(bundle)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                return hit
            if not (bundle / "meta.json").exists():
                raise ModelError(f"bundle directory {bundle.as_posix()} is gone")
            if not allow_gpu and self._gpu_visible():
                raise GpuVisible("TensorFlow sees a GPU: the desk server runs on the CPU (set CUDA_VISIBLE_DEVICES=-1 "
                                 "before the first TensorFlow import) or pass allow_gpu=True")
            predictor = self._loader(bundle)
            self._items = {k: v for k, v in self._items.items() if k[0] != key[0]}     # drop an older weights file
            self._items[key] = predictor
            return predictor

    def loaded(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class ModelRegistry:
    """The ``models`` table plus the predictor cache. ``runs_root`` is the engine's run store (dev scores)."""

    def __init__(self, db: DeskDb, *, runs_root: Union[str, Path, None] = None, home: Union[str, Path, None] = None,
                 cache: Optional[PredictorCache] = None, clock: Callable[[], float] = time.time) -> None:
        self.repo = RowRepo(db, "models", "id")
        self.runs_root = Path(runs_root) if runs_root is not None else PROJECT_ROOT / "runs"
        self.home = Path(home) if home is not None else DESK_HOME
        self.cache = cache or PredictorCache()
        self._clock = clock

    # ------------------------------------------------------------------ promote
    def _scores(self, run_dir: Path) -> Dict[str, Any]:
        from neural_trade.desk.lab.runs import LabRuns, RunNotFound

        labs = LabRuns(self.runs_root)
        try:
            row = labs._row(run_dir.name)        # in-package coupling, as runs.py has with longrun
        except RunNotFound:
            return {"run_id": run_dir.name, "split": None, "rank_eligible": False,
                    "note": "the run is not in the run store, so it has no recorded scores"}
        item = labs._item(row)
        out: Dict[str, Any] = {"run_id": row["run_id"], "scenario": row.get("scenario"), "split": item["split"],
                               "rank_eligible": item["rank_eligible"]}
        if item["split"] == "dev":
            out.update({k: v for k, v in item["key_numbers"].items() if k not in ("split", "rank_eligible")})
            try:
                from neural_trade.experiments.store import RunStore

                out["index_scores"] = RunStore(self.runs_root).index.scores(row["run_id"])
            except Exception:         # a missing or locked index leaves the key numbers
                out["index_scores"] = {}
        else:
            out["note"] = "test-fold run: its numbers are not used for ranking and are not recorded here (D-020)"
        return out

    def promote(self, path: Union[str, Path], *, copy_to_home: bool = False, model_id: Optional[str] = None
                ) -> Dict[str, Any]:
        """Record a trained bundle; a repeated promote of the same id refreshes the row and keeps ``active_for``."""
        run_dir, bundle = _locate(path)
        facts = bundle_facts(bundle)
        mid = model_id or run_dir.name
        if copy_to_home:
            target = self.home / "models" / mid
            if target.exists():
                raise ModelError(f"{target.as_posix()} already exists")
            tmp = target.with_name(target.name + ".part")
            if tmp.exists():
                shutil.rmtree(tmp)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundle, tmp)
            tmp.replace(target)
            bundle = target
        try:
            run_meta = _read_json(run_dir / "meta.json") if (run_dir / "meta.json").exists() else {}
        except ModelError:
            run_meta = {}
        fingerprint = (run_meta.get("dataset") or {}).get("sha256") if isinstance(run_meta.get("dataset"), dict) else None
        old = self.repo.get(mid) or {}
        row = {"id": mid, "run_dir": run_dir.as_posix(), "bundle_dir": bundle.as_posix(), **facts,
               "fingerprint": fingerprint, "dev_scores": self._scores(run_dir),
               "promoted_ms": int(self._clock() * 1000), "active_for": old.get("active_for") or []}
        self.repo.save(row)
        return row

    # ------------------------------------------------------------------ read
    def get(self, model_id: str) -> Dict[str, Any]:
        row = self.repo.get(model_id)
        if row is None:
            raise ModelError(f"no model {model_id!r}")
        return row

    def list(self) -> List[Dict[str, Any]]:
        return self.repo.list()

    # ------------------------------------------------------------------ activate
    def activate(self, model_id: str, symbol: Optional[str] = None, tf: Optional[int] = None, *,
                 force: bool = False) -> Dict[str, Any]:
        """Make ``model_id`` the model for (symbol, tf); the previous one for that pair is deactivated. A pair that
        differs from the model's own (symbol, tf) is refused unless ``force`` (the horizons would mean other time)."""
        row = self.get(model_id)
        symbol = row["symbol"] if symbol is None else str(symbol)
        tf = int(row["tf"] if tf is None else tf)
        if (symbol, tf) != (row["symbol"], int(row["tf"])) and not force:
            raise ModelError(f"model {model_id} was trained for {row['symbol']} {row['tf']}-minute bars, not "
                             f"{symbol} {tf}")
        pair = {"symbol": symbol, "tf": tf}
        for other in self.repo.list():
            if other["id"] == model_id:
                continue
            held = other.get("active_for") or []
            if pair in held:
                self.repo.save({**other, "active_for": [a for a in held if a != pair]})
        mine = [a for a in (row.get("active_for") or []) if a != pair] + [pair]
        self.repo.save({**row, "active_for": mine})
        return self.get(model_id)

    def active(self, symbol: str, tf: int) -> Optional[Dict[str, Any]]:
        pair = {"symbol": str(symbol), "tf": int(tf)}
        for row in self.repo.list():
            if pair in (row.get("active_for") or []):
                return row
        return None

    # ------------------------------------------------------------------ predictor
    def predictor(self, model_id: str, *, allow_gpu: bool = False) -> Any:
        return self.cache.get(self.get(model_id)["bundle_dir"], allow_gpu=allow_gpu)
