"""Lossless safetensors restart for reservoir, carry, clock, and parity state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyCarries, EconomyState

SNAPSHOT_SCHEMA = "sirrobin.economy.snapshot.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_snapshot(path: Path, state: EconomyState, config: EconomyConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "nd_q": state.nd_q.detach().cpu().contiguous(),
        "bp_q": state.bp_q.detach().cpu().contiguous(),
        "bd_q": state.bd_q.detach().cpu().contiguous(),
        "bm_q": state.bm_q.detach().cpu().contiguous(),
        "step": state.step.detach().cpu().contiguous(),
        "time_s": state.time_s.detach().cpu().contiguous(),
        "buffer_parity": state.buffer_parity.detach().cpu().contiguous(),
    }
    tensors.update(
        {
            f"carry.{field.name}": getattr(state.carries, field.name).detach().cpu().contiguous()
            for field in fields(state.carries)
        }
    )
    save_file(tensors, str(path), metadata={"schema": SNAPSHOT_SCHEMA, "config_hash": config.sha256()})
    metadata = {
        "schema": SNAPSHOT_SCHEMA,
        "config": json.loads(config.canonical_json()),
        "config_hash": config.sha256(),
        "step": int(state.step.item()),
        "time_s": float(state.time_s.item()),
        "buffer_parity": int(state.buffer_parity.item()),
        "tensor_sha256": _sha256(path),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_snapshot(path: Path, *, device: torch.device | str = "cpu") -> tuple[EconomyState, EconomyConfig]:
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unexpected economy snapshot schema")
    config = EconomyConfig(**metadata["config"])
    config.validate()
    if metadata["config_hash"] != config.sha256():
        raise ValueError("snapshot config hash mismatch")
    if metadata.get("tensor_sha256") != _sha256(path):
        raise ValueError("snapshot tensor hash mismatch")
    tensors = load_file(str(path), device=str(device))
    carries = EconomyCarries(
        **{field.name: tensors[f"carry.{field.name}"] for field in fields(EconomyCarries)}
    )
    state = EconomyState(
        tensors["nd_q"],
        tensors["bp_q"],
        tensors["bd_q"],
        tensors["bm_q"],
        carries,
        tensors["step"],
        tensors["time_s"],
        tensors["buffer_parity"],
    )
    state.validate(config)
    return state, config
