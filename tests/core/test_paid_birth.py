"""Focused contract for paid, atomic exact-clone birth."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from sirrobin.core.reproduction import BirthConfig, attempt_exact_clone_birth
from tools.run_world import _build_fixture_world


def _world(*, capacity: int = 3, live: int = 1, parent_reserve_q: int = 2_000):
    world = _build_fixture_world(
        bodies=capacity,
        live_bodies=live,
        reserve_q_per_creature=parent_reserve_q,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    return world


def _authoritative_snapshot(world):
    return (
        tuple(getattr(world.genotype, field.name).clone() for field in fields(world.genotype)),
        world.creature_material.structure_q.clone(),
        world.creature_material.reserve_q.clone(),
        tuple(carry.clone() for carry in world.creature_material.carries),
        tuple(getattr(world.live_state, field.name).clone() for field in fields(world.live_state)),
        world.next_stable_id.clone(),
    )


def _assert_authoritative_snapshot(world, expected) -> None:
    genotype, structure, reserve, carries, live_state, next_id = expected
    assert all(
        torch.equal(getattr(world.genotype, field.name), value)
        for field, value in zip(fields(world.genotype), genotype, strict=True)
    )
    assert torch.equal(world.creature_material.structure_q, structure)
    assert torch.equal(world.creature_material.reserve_q, reserve)
    assert all(
        torch.equal(actual, value)
        for actual, value in zip(world.creature_material.carries, carries, strict=True)
    )
    assert all(
        torch.equal(getattr(world.live_state, field.name), value)
        for field, value in zip(fields(world.live_state), live_state, strict=True)
    )
    assert torch.equal(world.next_stable_id, next_id)


def test_paid_birth_transfers_full_cost_and_rebuilds_an_exact_clone() -> None:
    world = _world()
    before = world.matter_totals()
    reserve_energy_before_j = float(
        world.creature_material.reserve_q.sum().item()
        * world.material_energy_config.reserve_j_per_q
    )
    parent_genotype = {
        field.name: getattr(world.genotype, field.name)[0, 0].clone()
        for field in fields(world.genotype)
        if field.name not in {"alive", "stable_id"}
    }
    parent_position = world.live_state.position_enu_m[0, 0].clone()
    parent_yaw = world.live_state.yaw_rad[0, 0].clone()

    report = attempt_exact_clone_birth(world, BirthConfig(initial_reserve_q=100))

    assert report.born is True
    assert report.reason is None
    assert report.parent_slot == 0
    assert report.child_slot == 1
    assert report.parent_id == 1
    assert report.child_id == 2
    assert report.structure_q == 1_000
    assert report.initial_reserve_q == 100
    assert report.total_debit_q == 1_100
    assert report.parent_reserve_before_q == 2_000
    assert report.parent_reserve_after_q == 900
    assert report.construction_heat_j == 450.0
    reserve_energy_after_j = float(
        world.creature_material.reserve_q.sum().item()
        * world.material_energy_config.reserve_j_per_q
    )
    assert reserve_energy_before_j - reserve_energy_after_j == (
        report.construction_heat_j
    )

    assert world.genotype.alive.tolist() == [[True, True, False]]
    assert world.body.alive.tolist() == [[True, True, False]]
    assert int(world.genotype.stable_id[0, 1]) == 2
    assert int(world.body.stable_id[0, 1]) == 2
    for name, expected in parent_genotype.items():
        assert torch.equal(getattr(world.genotype, name)[0, 1], expected), name
    assert torch.equal(world.body.seg_mask[0, 1], world.body.seg_mask[0, 0])
    assert torch.equal(world.body.mass_sim[0, 1], world.body.mass_sim[0, 0])

    assert world.creature_material.structure_q.tolist() == [[1_000, 1_000, 0]]
    assert world.creature_material.reserve_q.tolist() == [[900, 100, 0]]
    assert all(float(carry[0, 1]) == 0.0 for carry in world.creature_material.carries)
    assert torch.equal(world.live_state.position_enu_m[0, 1], parent_position)
    assert world.live_state.velocity_rel_water_enu_m_s[0, 1].tolist() == [0.0, 0.0, 0.0]
    assert torch.equal(world.live_state.yaw_rad[0, 1], parent_yaw)
    assert float(world.live_state.yaw_momentum_kg_m2_s[0, 1]) == 0.0
    assert float(world.live_state.gait_time_s[0, 1]) == 0.0
    assert world.live_state.heading_initialized[0, 1].item() is False
    assert world.next_stable_id.tolist() == [3]
    assert world.close_matter_step(before).books_closed.tolist() == [True]


def test_insufficient_reserve_refuses_without_allocating_or_mutating() -> None:
    world = _world(parent_reserve_q=1_099)
    before = _authoritative_snapshot(world)

    report = attempt_exact_clone_birth(world, BirthConfig(initial_reserve_q=100))

    assert report.born is False
    assert report.reason == "insufficient_reserve"
    assert report.child_slot is None
    assert report.child_id is None
    assert report.total_debit_q == 0
    assert report.construction_heat_j == 0.0
    _assert_authoritative_snapshot(world, before)


def test_slot_exhaustion_is_a_nonmutating_event_not_a_population_repair() -> None:
    world = _world(capacity=1, live=1)
    before = _authoritative_snapshot(world)

    report = attempt_exact_clone_birth(world, BirthConfig(initial_reserve_q=100))

    assert report.born is False
    assert report.reason == "slot_exhausted"
    assert report.child_slot is None
    assert report.child_id is None
    _assert_authoritative_snapshot(world, before)


def test_failed_child_development_is_preflighted_before_any_commit(monkeypatch) -> None:
    world = _world()
    before = _authoritative_snapshot(world)

    def reject_development(_candidate):
        raise RuntimeError("forced child development rejection")

    monkeypatch.setattr("sirrobin.core.reproduction.develop", reject_development)

    with pytest.raises(RuntimeError, match="forced child development rejection"):
        attempt_exact_clone_birth(world, BirthConfig(initial_reserve_q=100))

    _assert_authoritative_snapshot(world, before)


def test_first_free_slot_is_deterministic_and_historical_ids_are_not_reused() -> None:
    world = _world(capacity=4, live=4, parent_reserve_q=4_000)
    # Retire three real IDs and return their material. The allocator was bound
    # while those organisms were live, so reusing their now-free slots cannot
    # reuse their historical identities.
    for slot in (1, 2, 3):
        returned_q = int(
            world.creature_material.structure_q[0, slot]
            + world.creature_material.reserve_q[0, slot]
        )
        world.economy_state.nd_q[0, 0, 0, 0] += returned_q
        world.creature_material.structure_q[0, slot] = 0
        world.creature_material.reserve_q[0, slot] = 0
        for carry in world.creature_material.carries:
            carry[0, slot] = 0.0
        world.genotype.alive[0, slot] = False
    world.rebuild_body()
    # Make slot 2 observably dirty but inactive; a correct birth must choose slot 1
    # and initialize it rather than depending on inactive buffer contents.
    world.live_state.position_enu_m[0, 2] = torch.tensor([8.0, 7.0, 0.0])
    world.live_state.yaw_rad[0, 2] = 1.25

    first = attempt_exact_clone_birth(world, BirthConfig(initial_reserve_q=100))
    second = attempt_exact_clone_birth(
        world, BirthConfig(initial_reserve_q=100), parent_slot=0
    )

    assert first.born and second.born
    assert (first.child_slot, second.child_slot) == (1, 2)
    assert (first.child_id, second.child_id) == (5, 6)
    assert world.next_stable_id.tolist() == [7]
    assert torch.equal(
        world.live_state.position_enu_m[0, 2],
        world.live_state.position_enu_m[0, 0],
    )


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_birth_config_rejects_invalid_initial_reserve(value) -> None:
    error = TypeError if isinstance(value, (float, bool)) else ValueError
    with pytest.raises(error):
        BirthConfig(initial_reserve_q=value)


def test_birth_rejects_an_inactive_or_structureless_parent() -> None:
    world = _world()
    with pytest.raises(ValueError, match="parent must be alive"):
        attempt_exact_clone_birth(
            world, BirthConfig(initial_reserve_q=100), parent_slot=1
        )

    world.creature_material.structure_q[0, 0] = 0
    with pytest.raises(ValueError, match="positive structure"):
        attempt_exact_clone_birth(world, BirthConfig(initial_reserve_q=100))
