"""Lossless JSONL telemetry with explicit provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunManifest:
    config_hash: str
    corpus_hash: str
    device: str
    dtype: str
    torch_version: str
    hardware: str
    seed: int = 0


class TelemetryWriter:
    def __init__(self, path: Path, manifest: RunManifest):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8", newline="\n")
        self.write("manifest", asdict(manifest))

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        self._fh.write(json.dumps({"kind": kind, "payload": payload}, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
