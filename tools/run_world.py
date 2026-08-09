"""Run the existing composed SirRobin world without Unity.

This deliberately emits human-readable operational output, not a persistence or
observability schema.  Its purpose is to make the baseline world runnable and its
current multi-rate cost visible before new biology is added.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from sirrobin.core.runner import HeadlessRunner, WorldSchedule
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig

FIXTURE = Path(__file__).resolve().parents[1] / "oracle/fixtures/live/donor_development_live.json"
FIELD_NAMES = ("ND", "BP", "BD", "BM")


@dataclass(frozen=True, slots=True)
class WorldRunReport:
    requested_sim_time_s: float
    sim_time_s: float
    economy_steps: int
    mechanics_steps: int
    mechanics_steps_per_economy_step: int
    shipped_mechanics_steps_per_economy_step: int
    population: int
    initial_fields_q: tuple[int, int, int, int]
    final_fields_q: tuple[int, int, int, int]
    books_closed: bool
    gait_time_min_s: float
    gait_time_max_s: float
    positions_sample_enu_m: tuple[tuple[float, float, float], ...]
    setup_wall_time_s: float
    advance_wall_time_s: float

    @property
    def total_wall_time_s(self) -> float:
        return self.setup_wall_time_s + self.advance_wall_time_s

    @property
    def sim_seconds_per_wall_second(self) -> float:
        return self.sim_time_s / self.advance_wall_time_s


def _field_totals_q(state: EconomyState) -> tuple[int, int, int, int]:
    return tuple(int(reservoir.sum(dtype=torch.int64).item()) for reservoir in state.reservoirs)


def _build_fixture_world(*, bodies: int, device: torch.device) -> HeadlessWorld:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    swimmer = next(row for row in rows if row["id"] == "swimmer")
    genotype = GenotypeBatch.from_donor_rows(
        [swimmer] * bodies,
        dtype=torch.float64,
        device=device,
    )
    economy_config = replace(
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
    economy_state = EconomyState.zeros(economy_config, device=device)
    economy_state.nd_q.fill_(10_000_000)
    economy_state.bp_q.fill_(1_000_000)
    economy_state.bd_q[..., 0] = 500_000
    lead = (1, bodies)
    return HeadlessWorld(
        genotype=genotype,
        fluid=FluidSample(
            torch.full(lead, 1000.0, dtype=torch.float64, device=device),
            torch.zeros((*lead, 3), dtype=torch.float64, device=device),
        ),
        live_config=LiveLocomotionConfig(),
        economy_state=economy_state,
        economy_config=economy_config,
    )


def run_world(*, seconds: float, bodies: int, device_name: str) -> WorldRunReport:
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("seconds must be positive and finite")
    if bodies <= 0:
        raise ValueError("bodies must be positive")
    try:
        device = torch.device(device_name)
    except RuntimeError as error:
        raise ValueError(f"invalid device {device_name!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    setup_started = time.perf_counter()
    world = _build_fixture_world(bodies=bodies, device=device)
    runner = HeadlessRunner(world)
    interval_s = world.economy_config.dt_eco_s
    intervals = round(seconds / interval_s)
    if not math.isclose(seconds, intervals * interval_s, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"seconds must be an exact multiple of the fixture interval {interval_s:g}")
    initial_fields_q = _field_totals_q(world.economy_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_wall_time_s = time.perf_counter() - setup_started

    advance_started = time.perf_counter()
    books_closed = True
    mechanics_steps = 0
    for _ in range(intervals):
        tick = runner.advance()
        books_closed &= bool(tick.economy.books_closed.all())
        mechanics_steps += tick.mechanics_steps
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    advance_wall_time_s = time.perf_counter() - advance_started

    gait_time = world.live_state.gait_time_s
    positions = world.live_state.position_enu_m.reshape(-1, 3)
    sample = positions[: min(8, positions.shape[0])].detach().cpu().tolist()
    shipped_schedule = WorldSchedule.from_configs(LiveLocomotionConfig(), EconomyConfig())
    return WorldRunReport(
        requested_sim_time_s=seconds,
        sim_time_s=world.sim_time_s,
        economy_steps=int(world.economy_state.step.item()),
        mechanics_steps=mechanics_steps,
        mechanics_steps_per_economy_step=runner.schedule.mechanics_steps_per_economy_step,
        shipped_mechanics_steps_per_economy_step=(
            shipped_schedule.mechanics_steps_per_economy_step
        ),
        population=int(world.body.alive.sum().item()),
        initial_fields_q=initial_fields_q,
        final_fields_q=_field_totals_q(world.economy_state),
        books_closed=books_closed,
        gait_time_min_s=float(gait_time.min().item()),
        gait_time_max_s=float(gait_time.max().item()),
        positions_sample_enu_m=tuple(tuple(float(value) for value in row) for row in sample),
        setup_wall_time_s=setup_wall_time_s,
        advance_wall_time_s=advance_wall_time_s,
    )


def _fields_line(values: tuple[int, int, int, int]) -> str:
    return " ".join(f"{name}={value}" for name, value in zip(FIELD_NAMES, values, strict=True))


def format_report(report: WorldRunReport) -> str:
    positions = "\n".join(
        f"  {index}: ({east:.9g}, {north:.9g}, {up:.9g})"
        for index, (east, north, up) in enumerate(report.positions_sample_enu_m)
    )
    return "\n".join(
        (
            "SirRobin composed-world run (operational output; not a stable schema)",
            f"requested simulated time s: {report.requested_sim_time_s:g}",
            f"actual simulated time s: {report.sim_time_s:g}",
            f"economy steps: {report.economy_steps}",
            f"mechanics steps: {report.mechanics_steps}",
            f"mechanics steps / economy step: {report.mechanics_steps_per_economy_step}",
            "shipped mechanics steps / economy step: "
            f"{report.shipped_mechanics_steps_per_economy_step}",
            f"population: {report.population}",
            f"initial field totals q: {_fields_line(report.initial_fields_q)}",
            f"final field totals q: {_fields_line(report.final_fields_q)}",
            f"initial total q: {sum(report.initial_fields_q)}",
            f"final total q: {sum(report.final_fields_q)}",
            f"exact books closed: {'yes' if report.books_closed else 'no'}",
            f"mechanics clock range s: {report.gait_time_min_s:g} .. {report.gait_time_max_s:g}",
            f"positions sample ENU m ({len(report.positions_sample_enu_m)}/{report.population}):",
            positions,
            f"setup wall time s: {report.setup_wall_time_s:.6f}",
            f"advance wall time s: {report.advance_wall_time_s:.6f}",
            f"total wall time s: {report.total_wall_time_s:.6f}",
            "simulated seconds / wall second: "
            f"{report.sim_seconds_per_wall_second:.6f} (advance only)",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0.1)
    parser.add_argument("--bodies", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args(argv)
    try:
        report = run_world(
            seconds=arguments.seconds,
            bodies=arguments.bodies,
            device_name=arguments.device,
        )
    except ValueError as error:
        parser.error(str(error))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
