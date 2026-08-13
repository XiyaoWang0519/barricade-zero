"""Durable metadata and generation records for resumable training runs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_CHECKPOINT_PATTERN = re.compile(r"generation_(\d+)\.pt$")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_checkpoint(checkpoint_dir: str | Path) -> Path:
    root = Path(checkpoint_dir)
    matches = []
    for path in root.glob("generation_*.pt"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            matches.append((int(match.group(1)), path))
    if not matches:
        raise FileNotFoundError(f"no generation checkpoints found in {root}")
    return max(matches, key=lambda item: item[0])[1]


class RunLedger:
    def __init__(self, checkpoint_dir: str | Path, config: Mapping[str, Any]) -> None:
        self.root = Path(checkpoint_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "run.json"
        self.generations_path = self.root / "generations.jsonl"
        self.config = dict(config)
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("config") != self.config:
                raise ValueError(
                    f"training configuration does not match existing run in {self.root}"
                )
            return
        manifest = {
            "schema_version": 1,
            "created_at": _timestamp(),
            "config": self.config,
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.manifest_path)

    def record(self, summary: Mapping[str, Any]) -> None:
        record = dict(summary)
        record["recorded_at"] = _timestamp()
        with self.generations_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
