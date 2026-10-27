"""Small, dependency-light primitives for reproducible experiment runs.

The module intentionally does not configure the root logger.  This lets a
notebook, a CLI, and test suites coexist without duplicate log handlers.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json_dump(payload: Mapping[str, Any], path: os.PathLike | str, *, indent: int = 2) -> Path:
    """Atomically write JSON, so an interrupted run never leaves partial state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=indent, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


class JsonlFormatter(logging.Formatter):
    """Emit one structured event per line while retaining ordinary logging."""
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "event", None)
        if isinstance(extra, Mapping):
            event.update(extra)
        return json.dumps(event, default=_json_default, sort_keys=True)


def capture_system_info() -> Dict[str, Any]:
    """Return serializable environment details without failing on optional torch."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def estimate_runtime(*, epochs: int, batches_per_epoch: int, seconds_per_batch: float,
                     validation_batches: int = 0, seconds_per_validation_batch: Optional[float] = None) -> Dict[str, float]:
    """Estimate training wall-clock time from a short measured warm-up."""
    if min(epochs, batches_per_epoch, validation_batches) < 0 or seconds_per_batch < 0:
        raise ValueError("counts and timings must be non-negative")
    validation_seconds = seconds_per_batch if seconds_per_validation_batch is None else seconds_per_validation_batch
    total = epochs * (batches_per_epoch * seconds_per_batch + validation_batches * validation_seconds)
    return {"seconds": float(total), "minutes": float(total / 60.0), "hours": float(total / 3600.0)}


@dataclass
class ExperimentRun:
    """Standard output layout with durable metadata, JSONL telemetry and checkpoints."""
    root: os.PathLike | str = "results/runs"
    name: str = "experiment"
    run_id: Optional[str] = None
    resume: bool = False
    path: Path = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = self.run_id or f"{self.name}-{stamp}"
        self.path = Path(self.root) / self.run_id
        if self.path.exists() and not self.resume:
            raise FileExistsError(f"Run already exists: {self.path}; use resume=True")
        for directory in (self.path, self.path / "checkpoints", self.path / "figures", self.path / "artifacts"):
            directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"experiment.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.FileHandler(self.path / "events.jsonl", encoding="utf-8")
            handler.setFormatter(JsonlFormatter())
            self.logger.addHandler(handler)
        if not self.resume or not (self.path / "system.json").exists():
            atomic_json_dump(capture_system_info(), self.path / "system.json")

    @property
    def checkpoint_path(self) -> Path:
        return self.path / "checkpoints" / "last.pt"

    def save_config(self, config: Mapping[str, Any]) -> Path:
        return atomic_json_dump(config, self.path / "config.json")

    def save_state(self, state: Mapping[str, Any], filename: str = "state.json") -> Path:
        return atomic_json_dump(state, self.path / filename)

    def load_state(self, filename: str = "state.json", default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state_path = self.path / filename
        if not state_path.exists():
            return {} if default is None else dict(default)
        with state_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def event(self, message: str, **fields: Any) -> None:
        self.logger.info(message, extra={"event": fields})

    def save_checkpoint(self, payload: Mapping[str, Any], filename: str = "last.pt") -> Path:
        """Atomically persist a torch checkpoint, importing torch only when required."""
        import torch
        target = self.path / "checkpoints" / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(dict(payload), temporary)
        if target.exists():
            shutil.copy2(target, target.with_name(f"previous_{target.name}"))
        os.replace(temporary, target)
        return target

    def load_checkpoint(self, filename: str = "last.pt", map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        import torch
        target = self.path / "checkpoints" / filename
        candidates = (target, target.with_name(f"previous_{target.name}"))
        saw_candidate = False
        for candidate in candidates:
            if not candidate.exists():
                continue
            saw_candidate = True
            try:
                return torch.load(candidate, map_location=map_location, weights_only=False)
            except Exception as exc:
                self.event("checkpoint_rejected", path=str(candidate), error=repr(exc))
        if saw_candidate:
            raise RuntimeError("No valid checkpoint remains; refusing a silent restart")
        return None
