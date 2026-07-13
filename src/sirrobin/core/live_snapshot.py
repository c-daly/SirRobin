"""Lossless genotype plus live-state restart; DevelopedBody is always regenerated."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import DevelopedBody, LiveState
from sirrobin.physics.live_config import LiveLocomotionConfig

SNAPSHOT_SCHEMA = "sirrobin.live.snapshot.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_live_snapshot(
    path: Path,
    genotype: GenotypeBatch,
    state: LiveState,
    config: LiveLocomotionConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        **{
            f"genotype.{field.name}": getattr(genotype, field.name).detach().cpu().contiguous()
            for field in fields(genotype)
        },
        **{
            f"state.{field.name}": getattr(state, field.name).detach().cpu().contiguous()
            for field in fields(state)
        },
    }
    save_file(tensors, str(path), metadata={"schema": SNAPSHOT_SCHEMA, "config_hash": config.sha256()})
    metadata = {
        "schema": SNAPSHOT_SCHEMA,
        "config": json.loads(config.canonical_json()),
        "config_hash": config.sha256(),
        "tensor_sha256": _sha256(path),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_live_snapshot(
    path: Path, *, device: torch.device | str = "cpu"
) -> tuple[GenotypeBatch, DevelopedBody, LiveState, LiveLocomotionConfig]:
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unexpected live snapshot schema")
    config = LiveLocomotionConfig(**metadata["config"])
    config.validate()
    if metadata.get("config_hash") != config.sha256():
        raise ValueError("live snapshot config hash mismatch")
    if metadata.get("tensor_sha256") != _sha256(path):
        raise ValueError("live snapshot tensor hash mismatch")
    tensors = load_file(str(path), device=str(device))
    genotype_names = {field.name for field in fields(GenotypeBatch)}
    state_names = {field.name for field in fields(LiveState)}
    expected = {f"genotype.{name}" for name in genotype_names} | {
        f"state.{name}" for name in state_names
    }
    if set(tensors) != expected:
        missing = sorted(expected - set(tensors))
        extra = sorted(set(tensors) - expected)
        raise ValueError(f"live snapshot tensor schema mismatch: missing={missing}, extra={extra}")
    genotype = GenotypeBatch(**{name: tensors[f"genotype.{name}"] for name in genotype_names})
    genotype.validate()
    state = LiveState(**{name: tensors[f"state.{name}"] for name in state_names})
    return genotype, develop(genotype), state, config
