"""Exact field-plus-creature material authority for the first lifecycle."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sirrobin.core.material import CreatureMaterialState, MaterialEnergyConfig
from sirrobin.core.runner import HeadlessRunner
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")
TEST_ENERGY = MaterialEnergyConfig(0.50, 0.45)


def _world(
    *,
    structure_q: tuple[int, ...] = (1_000, 1_000),
    reserve_q: tuple[int, ...] = (500, 250),
    second_alive: bool = True,
    material: CreatureMaterialState | None = None,
    energy: object = TEST_ENERGY,
) -> HeadlessWorld:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    swimmer = next(row for row in rows if row["id"] == "swimmer")
    if len(structure_q) != len(reserve_q):
        raise ValueError("test material rows must align")
    capacity = len(structure_q)
    genotype = GenotypeBatch.from_donor_rows(
        [swimmer] * capacity, dtype=torch.float64
    )
    genotype.alive[0, 1] = second_alive
    config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=4,
        lx_m=10.0,
        ly_m=10.0,
        lz_m=20.0,
        dt_eco_s=0.1,
        remin_floor_s=1.0e-4,
    )
    economy = EconomyState.zeros(config)
    economy.nd_q.fill_(10_000_000)
    economy.bp_q.fill_(1_000_000)
    economy.bd_q[..., 0] = 500_000
    if material is None:
        carry = torch.zeros((1, capacity), dtype=torch.float64)
        material = CreatureMaterialState(
            structure_q=torch.tensor([structure_q], dtype=torch.int64),
            reserve_q=torch.tensor([reserve_q], dtype=torch.int64),
            intake_carry_mol=carry.clone(),
            assimilation_carry_q=carry.clone(),
        )
    return HeadlessWorld(
        genotype=genotype,
        fluid=FluidSample(
            torch.full((1, capacity), 1000.0, dtype=torch.float64),
            torch.zeros((1, capacity, 3), dtype=torch.float64),
        ),
        live_config=LiveLocomotionConfig(),
        economy_state=economy,
        economy_config=config,
        creature_material_state=material,
        material_energy_config=energy,
    )


def test_complete_tick_closes_raw_field_plus_creature_sums_exactly() -> None:
    world = _world()
    runner = HeadlessRunner(world)
    field_before = sum(
        int(reservoir.sum(dtype=torch.int64).item())
        for reservoir in world.economy_state.reservoirs
    )
    structure_before = int(world.creature_material.structure_q.sum().item())
    reserve_before = int(world.creature_material.reserve_q.sum().item())
    expected = field_before + structure_before + reserve_before

    tick = runner.advance()

    field_after = sum(
        int(reservoir.sum(dtype=torch.int64).item())
        for reservoir in world.economy_state.reservoirs
    )
    assert tick.matter.field_before_q.tolist() == [field_before]
    assert tick.matter.structure_before_q.tolist() == [structure_before]
    assert tick.matter.reserve_before_q.tolist() == [reserve_before]
    assert tick.matter.total_before_q.tolist() == [expected]
    assert tick.matter.field_after_q.tolist() == [field_after]
    assert tick.matter.structure_after_q.tolist() == [structure_before]
    assert tick.matter.reserve_after_q.tolist() == [reserve_before]
    assert tick.matter.total_after_q.tolist() == [expected]
    assert tick.matter.expected_total_q.tolist() == [expected]
    assert tick.matter.books_closed.tolist() == [True]
    assert torch.equal(world.creature_material.structure_q, torch.tensor([[1_000, 1_000]]))
    assert torch.equal(world.creature_material.reserve_q, torch.tensor([[500, 250]]))


@pytest.mark.parametrize("invalid", [None, "not-an-energy-config"])
def test_world_requires_runtime_material_energy_authority(invalid: object) -> None:
    with pytest.raises(TypeError, match="MaterialEnergyConfig"):
        _world(energy=invalid)


def test_explicit_zero_material_state_has_population_shape() -> None:
    world = _world(structure_q=(0, 0), reserve_q=(0, 0))

    assert world.creature_material.structure_q.shape == world.body.alive.shape
    assert world.creature_material.reserve_q.shape == world.body.alive.shape
    assert world.creature_material.structure_q.dtype == torch.int64
    assert world.creature_material.reserve_q.dtype == torch.int64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("structure_q", torch.zeros((1, 2), dtype=torch.float64), "int64"),
        ("reserve_q", torch.zeros((2,), dtype=torch.int64), "shape"),
        ("structure_q", torch.tensor([[1, -1]], dtype=torch.int64), "nonnegative"),
    ],
)
def test_malformed_creature_material_is_rejected(
    field: str, value: torch.Tensor, message: str
) -> None:
    values = {
        "structure_q": torch.tensor([[1, 1]], dtype=torch.int64),
        "reserve_q": torch.tensor([[1, 1]], dtype=torch.int64),
        "intake_carry_mol": torch.zeros((1, 2), dtype=torch.float64),
        "assimilation_carry_q": torch.zeros((1, 2), dtype=torch.float64),
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        _world(material=CreatureMaterialState(**values))


def test_inactive_capacity_cannot_hide_material() -> None:
    material = CreatureMaterialState(
        structure_q=torch.tensor([[1, 1]], dtype=torch.int64),
        reserve_q=torch.tensor([[0, 0]], dtype=torch.int64),
        intake_carry_mol=torch.zeros((1, 2), dtype=torch.float64),
        assimilation_carry_q=torch.zeros((1, 2), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="inactive"):
        _world(material=material, second_alive=False)


def test_whole_world_inventory_respects_the_safe_exact_reduction_bound() -> None:
    material = CreatureMaterialState(
        structure_q=torch.tensor([[10**15, 0]], dtype=torch.int64),
        reserve_q=torch.zeros((1, 2), dtype=torch.int64),
        intake_carry_mol=torch.zeros((1, 2), dtype=torch.float64),
        assimilation_carry_q=torch.zeros((1, 2), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="whole-world inventory"):
        _world(material=material)


def test_creature_mint_is_detected_and_arrests_the_runner() -> None:
    runner = HeadlessRunner(_world())
    runner.world.creature_material.reserve_q[0, 0] += 1

    with pytest.raises(RuntimeError, match="whole-world nutrient books do not close"):
        runner.advance()

    arrested_at = runner.world.sim_time_s
    with pytest.raises(RuntimeError, match="not resumable"):
        runner.advance()
    assert runner.world.sim_time_s == arrested_at


def test_int64_wrap_cannot_disguise_a_post_initialization_mint() -> None:
    runner = HeadlessRunner(
        _world(structure_q=(0, 0, 0), reserve_q=(0, 0, 0))
    )
    world = runner.world
    expected = int(world.expected_matter_total_q.item())
    q = 2**62 - 1
    world.creature_material.structure_q[0] = torch.tensor([q, q, q])
    world.creature_material.reserve_q[0] = torch.tensor([q, 4, 0])

    raw_python_total = sum(
        int(value)
        for reservoir in world.economy_state.reservoirs
        for value in reservoir.reshape(-1).tolist()
    ) + sum(
        int(value)
        for reservoir in world.creature_material.reservoirs
        for value in reservoir.reshape(-1).tolist()
    )
    assert raw_python_total == 2**64 + expected
    assert int(world.matter_totals().total_q.item()) == expected

    with pytest.raises(RuntimeError, match="whole-world nutrient books do not close"):
        runner.advance()


@pytest.mark.parametrize(
    "corruption",
    ["fractional_reserve", "reserve_shape", "field_dtype", "invalid_intake_carry"],
)
def test_runtime_reservoir_schema_corruption_arrests_before_reduction(
    corruption: str,
) -> None:
    runner = HeadlessRunner(_world())
    world = runner.world
    if corruption == "fractional_reserve":
        world.creature_material.reserve_q = torch.tensor([[500.0, 250.5]])
    elif corruption == "reserve_shape":
        world.creature_material.reserve_q = torch.tensor([500, 250], dtype=torch.int64)
    elif corruption == "invalid_intake_carry":
        world.creature_material.intake_carry_mol[0, 0] = float("nan")
    else:
        world.economy_state.bp_q = world.economy_state.bp_q.to(torch.float64)

    with pytest.raises(RuntimeError, match="invalid raw reservoir state"):
        runner.advance()

    assert world.sim_time_s == 0.0
    with pytest.raises(RuntimeError, match="not resumable"):
        runner.advance()


def test_equal_field_to_creature_transfer_closes_only_at_whole_world_scope() -> None:
    runner = HeadlessRunner(_world())
    world = runner.world
    field_before = world.economy_state.total_per_world().clone()
    world.economy_state.bp_q[0, 0, 0, 0] -= 10
    world.creature_material.reserve_q[0, 0] += 10

    tick = runner.advance()

    assert tick.economy.total_before_q.tolist() == (field_before - 10).tolist()
    assert tick.economy.books_closed.tolist() == [True]
    assert tick.matter.total_before_q.tolist() == world.expected_matter_total_q.tolist()
    assert tick.matter.books_closed.tolist() == [True]
