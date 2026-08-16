"""Versioned, lossless checkpoints for accepted device-runtime state."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyCarries, EconomyState
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.organisms.behavior import BehaviorConfig
from sirrobin.organisms.development import DevelopmentConfig, DevelopmentState
from sirrobin.organisms.feeding import FeedingConfig
from sirrobin.organisms.metabolism import MetabolismConfig
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.organisms.mutation import MutationConfig
from sirrobin.organisms.state import PopulationState
from sirrobin.physics.contracts import DevelopedBody, LiveState
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.phase_response import PhaseWindowConfig
from sirrobin.runtime.config import (
    LivingRuntimeConfig,
    validate_living_runtime_config,
)
from sirrobin.runtime.state import LivingState, validate_living_state

RUNTIME_CHECKPOINT_SCHEMA = "sirrobin.runtime.checkpoint.v2"
RUNTIME_CHECKPOINT_LAYOUT_SHA256 = (
    "b8a490dfc3e09b33bd43f235b65a9e6cc949a9d1328941039c8c3f7bc7dc3a87"
)

_CONFIG_COMPONENTS = {
    "economy": EconomyConfig,
    "live": LiveLocomotionConfig,
    "motion": PhaseWindowConfig,
    "behavior": BehaviorConfig,
    "feeding": FeedingConfig,
    "metabolism": MetabolismConfig,
    "mortality": MortalityConfig,
    "mutation": MutationConfig,
    "development": DevelopmentConfig,
}
_CONFIG_FIELDS = frozenset(
    (
        *_CONFIG_COMPONENTS,
        "child_initial_reserve_q",
        "birth_release_impulse_ns",
        "birth_separation_clearance_m",
    )
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _config_payload(config: LivingRuntimeConfig) -> dict[str, object]:
    payload = {
        name: asdict(getattr(config, name)) for name in _CONFIG_COMPONENTS
    }
    payload["child_initial_reserve_q"] = config.child_initial_reserve_q
    payload["birth_release_impulse_ns"] = config.birth_release_impulse_ns
    payload["birth_separation_clearance_m"] = (
        config.birth_separation_clearance_m
    )
    return payload


def _config_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    stored = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(stored.dtype).encode())
    digest.update(b"\0")
    digest.update(_canonical_json(list(stored.shape)).encode())
    digest.update(b"\0")
    byte_view = (stored.reshape(1) if stored.ndim == 0 else stored).view(
        torch.uint8
    )
    digest.update(byte_view.reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def _load_config(payload: object) -> LivingRuntimeConfig:
    if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
        raise ValueError("runtime checkpoint config fields differ from schema")
    restored: dict[str, Any] = {}
    for name, config_type in _CONFIG_COMPONENTS.items():
        component = payload[name]
        expected = {field.name for field in fields(config_type)}
        if not isinstance(component, dict) or set(component) != expected:
            raise ValueError(
                f"runtime checkpoint {name} config fields differ from schema"
            )
        restored[name] = config_type(**component)
    restored["child_initial_reserve_q"] = payload["child_initial_reserve_q"]
    restored["birth_release_impulse_ns"] = payload["birth_release_impulse_ns"]
    restored["birth_separation_clearance_m"] = payload[
        "birth_separation_clearance_m"
    ]
    config = LivingRuntimeConfig(**restored)
    validate_living_runtime_config(config)
    return config


def _flatten_tensors(
    prefix: str,
    value: object,
    tensors: dict[str, torch.Tensor],
) -> None:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"checkpoint component {prefix} must be a dataclass")
    for field in fields(value):
        item = getattr(value, field.name)
        key = f"{prefix}.{field.name}"
        if isinstance(item, torch.Tensor):
            # Identity views intentionally share storage in a live session.
            # Checkpoints preserve their equal values, while independent stored
            # tensors avoid making Python aliasing part of the public schema.
            tensors[key] = item.detach().cpu().contiguous().clone()
        elif is_dataclass(item) and not isinstance(item, type):
            _flatten_tensors(key, item, tensors)
        else:
            raise TypeError(f"checkpoint field {key} must be a tensor or dataclass")


def _tensor_fields(prefix: str, data_type: type[object]) -> set[str]:
    return {f"{prefix}.{field.name}" for field in fields(data_type)}


def _expected_tensor_fields() -> set[str]:
    expected = set()
    for prefix, data_type in (
        ("state.population", PopulationState),
        ("state.genotype", GenotypeBatch),
        ("state.body", DevelopedBody),
        ("state.development", DevelopmentState),
        ("state.motion", LiveState),
    ):
        expected.update(_tensor_fields(prefix, data_type))
    expected.update(
        f"state.economy.{field.name}"
        for field in fields(EconomyState)
        if field.name != "carries"
    )
    expected.update(_tensor_fields("state.economy.carries", EconomyCarries))
    expected.add("state.expected_matter_q")
    return expected


def _layout_sha256() -> str:
    layout = {
        "tensors": sorted(_expected_tensor_fields()),
        "config": {
            name: [field.name for field in fields(data_type)]
            for name, data_type in _CONFIG_COMPONENTS.items()
        },
        "config_root": sorted(_CONFIG_FIELDS),
    }
    return hashlib.sha256(_canonical_json(layout).encode()).hexdigest()


def _validate_schema_layout() -> None:
    if _layout_sha256() != RUNTIME_CHECKPOINT_LAYOUT_SHA256:
        raise RuntimeError(
            "runtime checkpoint layout changed without an explicit schema decision"
        )


def _restore_tensor_dataclass(
    data_type: type[Any],
    prefix: str,
    tensors: dict[str, torch.Tensor],
) -> Any:
    return data_type(
        **{
            field.name: tensors[f"{prefix}.{field.name}"]
            for field in fields(data_type)
        }
    )


def _restore_state(tensors: dict[str, torch.Tensor]) -> LivingState:
    actual = set(tensors)
    expected = _expected_tensor_fields()
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "runtime checkpoint tensor fields differ from schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    carries = _restore_tensor_dataclass(
        EconomyCarries,
        "state.economy.carries",
        tensors,
    )
    economy = EconomyState(
        nd_q=tensors["state.economy.nd_q"],
        bp_q=tensors["state.economy.bp_q"],
        bd_q=tensors["state.economy.bd_q"],
        bm_q=tensors["state.economy.bm_q"],
        carries=carries,
        step=tensors["state.economy.step"],
        time_s=tensors["state.economy.time_s"],
        buffer_parity=tensors["state.economy.buffer_parity"],
    )
    return LivingState(
        population=_restore_tensor_dataclass(
            PopulationState,
            "state.population",
            tensors,
        ),
        genotype=_restore_tensor_dataclass(
            GenotypeBatch,
            "state.genotype",
            tensors,
        ),
        body=_restore_tensor_dataclass(DevelopedBody, "state.body", tensors),
        development=_restore_tensor_dataclass(
            DevelopmentState,
            "state.development",
            tensors,
        ),
        motion=_restore_tensor_dataclass(LiveState, "state.motion", tensors),
        economy=economy,
        expected_matter_q=tensors["state.expected_matter_q"],
    )


def _temporary_path(path: Path, label: str) -> Path:
    file_descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f".{label}.tmp",
        dir=path.parent,
    )
    os.close(file_descriptor)
    return Path(name)


def _publish_checkpoint(staged: Path, target: Path) -> None:
    os.replace(staged, target)


def save_runtime_checkpoint(
    path: Path | str,
    state: LivingState,
    config: LivingRuntimeConfig,
) -> None:
    """Atomically publish one accepted runtime boundary as a checkpoint file."""

    target = Path(path)
    _validate_schema_layout()
    validate_living_runtime_config(config)
    validate_living_state(state, config.economy)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    _flatten_tensors("state", state, tensors)
    if set(tensors) != _expected_tensor_fields():
        raise RuntimeError("runtime state fields changed without a checkpoint schema change")
    config_payload = _config_payload(config)
    config_hash = _config_sha256(config_payload)
    tensor_hashes = {
        name: _tensor_sha256(value) for name, value in tensors.items()
    }
    tensor_temp = _temporary_path(target, "tensor")
    try:
        save_file(
            tensors,
            str(tensor_temp),
            metadata={
                "schema": RUNTIME_CHECKPOINT_SCHEMA,
                "layout_sha256": RUNTIME_CHECKPOINT_LAYOUT_SHA256,
                "config": _canonical_json(config_payload),
                "config_hash": config_hash,
                "tensor_sha256": _canonical_json(tensor_hashes),
                "step": str(int(state.economy.step.detach().cpu())),
                "time_s": repr(float(state.economy.time_s.detach().cpu())),
                "worlds": str(int(state.population.alive.shape[0])),
                "capacity": str(int(state.population.alive.shape[1])),
            },
        )
        _publish_checkpoint(tensor_temp, target)
    finally:
        tensor_temp.unlink(missing_ok=True)


def load_runtime_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[LivingState, LivingRuntimeConfig]:
    """Load and validate one complete runtime checkpoint onto ``device``."""

    source = Path(path)
    _validate_schema_layout()
    with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    expected_metadata = {
        "schema",
        "layout_sha256",
        "config",
        "config_hash",
        "tensor_sha256",
        "step",
        "time_s",
        "worlds",
        "capacity",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata:
        raise ValueError("runtime checkpoint metadata fields differ from schema")
    if metadata["schema"] != RUNTIME_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected runtime checkpoint schema")
    if metadata["layout_sha256"] != RUNTIME_CHECKPOINT_LAYOUT_SHA256:
        raise ValueError("runtime checkpoint layout hash mismatch")
    try:
        config_payload = json.loads(metadata["config"])
    except json.JSONDecodeError as error:
        raise ValueError("runtime checkpoint config metadata is invalid") from error
    config_hash = _config_sha256(config_payload)
    if metadata["config_hash"] != config_hash:
        raise ValueError("runtime checkpoint config hash mismatch")
    try:
        tensor_hashes = json.loads(metadata["tensor_sha256"])
    except json.JSONDecodeError as error:
        raise ValueError("runtime checkpoint tensor hashes are invalid") from error
    if not isinstance(tensor_hashes, dict) or set(tensor_hashes) != (
        _expected_tensor_fields()
    ):
        raise ValueError("runtime checkpoint tensor hash fields differ from schema")
    config = _load_config(config_payload)
    tensors = load_file(str(source), device=str(torch.device(device)))
    if set(tensors) != _expected_tensor_fields():
        raise ValueError("runtime checkpoint tensor fields differ from schema")
    if any(
        tensor_hashes[name] != _tensor_sha256(value)
        for name, value in tensors.items()
    ):
        raise ValueError("runtime checkpoint tensor hash mismatch")
    state = _restore_state(tensors)
    validate_living_state(state, config.economy)
    if int(metadata["step"]) != int(state.economy.step.detach().cpu()):
        raise ValueError("runtime checkpoint step metadata mismatch")
    if float(metadata["time_s"]) != float(state.economy.time_s.detach().cpu()):
        raise ValueError("runtime checkpoint time metadata mismatch")
    if int(metadata["worlds"]) != state.population.alive.shape[0]:
        raise ValueError("runtime checkpoint world-count metadata mismatch")
    if int(metadata["capacity"]) != state.population.alive.shape[1]:
        raise ValueError("runtime checkpoint capacity metadata mismatch")
    return state, config
