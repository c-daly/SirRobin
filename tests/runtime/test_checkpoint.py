from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import sirrobin.runtime.checkpoint as checkpoint_module
from sirrobin.physics.contracts import FluidSample
from sirrobin.runtime.checkpoint import load_runtime_checkpoint
from sirrobin.runtime.profile import (
    EVOLUTION_DEMO_RUNTIME_PROFILE,
    living_runtime_config_from_reference,
)
from sirrobin.runtime.reference_adapter import living_state_from_reference
from sirrobin.runtime.session import RuntimeSession
from tools.run_world import (
    LIVING_MATERIAL_ENERGY_CONFIG,
    _build_fixture_world,
)


def _assert_exact(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert tuple(left.shape) == tuple(right.shape)
        assert torch.equal(left.cpu(), right.cpu())
        return
    if is_dataclass(left) and not isinstance(left, type):
        assert type(left) is type(right)
        for field in fields(left):
            _assert_exact(getattr(left, field.name), getattr(right, field.name))
        return
    assert left == right


def _session_fixture(
    *,
    device: torch.device | str = "cpu",
) -> tuple[RuntimeSession, FluidSample]:
    target = torch.device(device)
    world = _build_fixture_world(
        bodies=3,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=target,
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    state = living_state_from_reference(world)
    config = living_runtime_config_from_reference(
        world,
        state,
        profile=EVOLUTION_DEMO_RUNTIME_PROFILE,
    )
    return (
        RuntimeSession(
            state,
            config,
            compile_motion=False,
            compile_domains=False,
            optimistic_motion=False,
            optimistic_feeding=False,
            optimistic_candidates=False,
        ),
        world.fluid,
    )


def test_checkpoint_restores_all_authoritative_state_and_exact_continuation(
    tmp_path: Path,
) -> None:
    session, fluid = _session_fixture()
    session.advance_autonomous_chunk(fluid, intervals=3)
    checkpoint = tmp_path / "living.safetensors"

    session.save_checkpoint(checkpoint)
    resumed = RuntimeSession.from_checkpoint(
        checkpoint,
        device="cpu",
        compile_motion=False,
        compile_domains=False,
        optimistic_motion=False,
        optimistic_feeding=False,
        optimistic_candidates=False,
    )

    assert resumed.config == session.config
    _assert_exact(resumed.state, session.state)
    uninterrupted = session.advance_autonomous_chunk(fluid, intervals=4)
    continued = resumed.advance_autonomous_chunk(fluid, intervals=4)
    _assert_exact(continued.state, uninterrupted.state)
    _assert_exact(continued.summary, uninterrupted.summary)
    assert bool(continued.last_interval.economy.ledger.books_closed.all())
    assert bool(continued.last_interval.matter.books_closed.all())


def test_checkpoint_rejects_tampered_tensor_bytes(tmp_path: Path) -> None:
    session, _ = _session_fixture()
    checkpoint = tmp_path / "living.safetensors"
    session.save_checkpoint(checkpoint)
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)

    with pytest.raises(ValueError, match="tensor hash mismatch"):
        load_runtime_checkpoint(checkpoint)


def test_interrupted_publication_preserves_the_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, fluid = _session_fixture()
    checkpoint = tmp_path / "living.safetensors"
    session.save_checkpoint(checkpoint)
    previous, _ = load_runtime_checkpoint(checkpoint)
    session.advance_autonomous_chunk(fluid, intervals=1)

    def interrupt(_staged: Path, _target: Path) -> None:
        raise OSError("simulated interruption before atomic publication")

    monkeypatch.setattr(checkpoint_module, "_publish_checkpoint", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        session.save_checkpoint(checkpoint)

    restored, _ = load_runtime_checkpoint(checkpoint)
    _assert_exact(restored, previous)
    assert not list(tmp_path.glob(".living.safetensors.*.tensor.tmp"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "sirrobin.runtime.checkpoint.v0", "schema"),
        ("config.mutation.seed", 99, "config hash mismatch"),
    ],
)
def test_checkpoint_rejects_tampered_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    session, _ = _session_fixture()
    checkpoint = tmp_path / "living.safetensors"
    session.save_checkpoint(checkpoint)
    tensors = load_file(str(checkpoint))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
    if field == "schema":
        metadata["schema"] = value
    else:
        config = json.loads(metadata["config"])
        config["mutation"]["seed"] = value
        canonical = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        )
        metadata["config"] = canonical
        # Leave the original binding in place: edited config must fail closed.
        assert metadata["config_hash"] != hashlib.sha256(
            canonical.encode()
        ).hexdigest()
    save_file(tensors, str(checkpoint), metadata=metadata)

    with pytest.raises(ValueError, match=message):
        load_runtime_checkpoint(checkpoint)


@pytest.mark.gpu
def test_checkpoint_can_restore_authoritative_state_to_cuda(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    session, _ = _session_fixture()
    checkpoint = tmp_path / "living.safetensors"
    session.save_checkpoint(checkpoint)

    state, config = load_runtime_checkpoint(checkpoint, device="cuda")

    assert state.population.alive.device.type == "cuda"
    assert state.economy.nd_q.device.type == "cuda"
    assert state.genotype.node_mask.device.type == "cuda"
    assert config == session.config
