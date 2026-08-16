#!/usr/bin/env python3
"""Measure local food state, autonomous effort, and physical paths."""

from __future__ import annotations

import argparse
import json
import math

import torch

from tools.runtime_unity import CAUSAL_RUNTIME_PROFILE, RuntimeUnityBackend
from tools.serve_unity import (
    ECONOMY_INTERVAL_S,
    INITIAL_BODIES,
    _build_server_world,
    _seed_visible_baseline,
)

INTERPRETATION = (
    "observed foraging requests and physical outcomes; direct paths, circles, "
    "overshoot, and failure are not pass/fail"
)


def _whole_intervals(duration_s: float) -> int:
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(duration_s)
        or duration_s <= 0.0
    ):
        raise ValueError("duration_s must be finite and positive")
    ratio = duration_s / ECONOMY_INTERVAL_S
    intervals = round(ratio)
    if not math.isclose(ratio, intervals, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("duration_s must contain whole authoritative intervals")
    return intervals


def _new_track(
    identity: int,
    position: torch.Tensor,
    yaw: float,
) -> dict[str, object]:
    return {
        "id": identity,
        "first_position_xy_m": position.tolist(),
        "last_position_xy_m": position.tolist(),
        "last_yaw_rad": yaw,
        "displacement_xy_m": [0.0, 0.0],
        "path_length_m": 0.0,
        "signed_yaw_change_rad": 0.0,
        "absolute_yaw_change_rad": 0.0,
        "behavior_intervals": 0,
        "food_gradient_intervals": 0,
        "locomoting_intervals": 0,
        "requested_effort_sum": 0.0,
        "feeding_debit_q": 0,
        "sampled_producer_sum_mol_m3": 0.0,
    }


def _observe_motion(
    tracks: dict[int, dict[str, object]],
    state,
    *,
    lx_m: float,
    ly_m: float,
) -> None:
    alive = state.population.alive[0].detach().cpu()
    stable_id = state.population.stable_id[0].detach().cpu()
    positions = state.motion.position_enu_m[0, :, :2].detach().cpu().to(torch.float64)
    yaw = state.motion.yaw_rad[0].detach().cpu().to(torch.float64)
    periods = torch.tensor([lx_m, ly_m], dtype=torch.float64)
    for slot in alive.nonzero().flatten().tolist():
        identity = int(stable_id[slot])
        position = positions[slot]
        angle = float(yaw[slot])
        if identity not in tracks:
            tracks[identity] = _new_track(identity, position, angle)
            continue
        track = tracks[identity]
        last_position = torch.tensor(track["last_position_xy_m"], dtype=torch.float64)
        delta = torch.remainder(position - last_position + 0.5 * periods, periods) - (0.5 * periods)
        displacement = torch.tensor(track["displacement_xy_m"], dtype=torch.float64) + delta
        yaw_delta = math.atan2(
            math.sin(angle - float(track["last_yaw_rad"])),
            math.cos(angle - float(track["last_yaw_rad"])),
        )
        track["last_position_xy_m"] = position.tolist()
        track["last_yaw_rad"] = angle
        track["displacement_xy_m"] = displacement.tolist()
        track["path_length_m"] = float(track["path_length_m"]) + float(torch.linalg.vector_norm(delta))
        track["signed_yaw_change_rad"] = float(track["signed_yaw_change_rad"]) + yaw_delta
        track["absolute_yaw_change_rad"] = float(track["absolute_yaw_change_rad"]) + abs(yaw_delta)


def _observe_behavior(tracks: dict[int, dict[str, object]], before, chunk) -> None:
    behavior = chunk.last_behavior
    if behavior is None:
        raise RuntimeError("autonomous diagnostic requires behavior output")
    alive = before.population.alive[0].detach().cpu()
    stable_id = before.population.stable_id[0].detach().cpu()
    gradient_present = behavior.horizontal_gradient_present[0].detach().cpu()
    locomoting = behavior.locomoting[0].detach().cpu()
    effort = behavior.requested_effort_fraction[0].detach().cpu()
    producer = behavior.sampled_producer_mol_m3[0].detach().cpu()
    feeding = chunk.last_interval.feeding.ledger.actual_debit_q[0].detach().cpu()
    for slot in alive.nonzero().flatten().tolist():
        identity = int(stable_id[slot])
        track = tracks[identity]
        track["behavior_intervals"] = int(track["behavior_intervals"]) + 1
        if bool(gradient_present[slot]):
            track["food_gradient_intervals"] = (
                int(track["food_gradient_intervals"]) + 1
            )
        if bool(locomoting[slot]):
            track["locomoting_intervals"] = int(track["locomoting_intervals"]) + 1
        track["requested_effort_sum"] = float(track["requested_effort_sum"]) + float(effort[slot])
        track["feeding_debit_q"] = int(track["feeding_debit_q"]) + int(feeding[slot])
        track["sampled_producer_sum_mol_m3"] = float(track["sampled_producer_sum_mol_m3"]) + float(
            producer[slot]
        )


def _finalize_track(track: dict[str, object], final_by_id: dict[int, dict[str, int]]) -> dict[str, object]:
    intervals = int(track.pop("behavior_intervals"))
    displacement = torch.tensor(track.pop("displacement_xy_m"), dtype=torch.float64)
    displacement_m = float(torch.linalg.vector_norm(displacement))
    path_length_m = float(track["path_length_m"])
    track.pop("last_yaw_rad")
    food_gradient_intervals = int(track.pop("food_gradient_intervals"))
    locomoting_intervals = int(track.pop("locomoting_intervals"))
    requested_effort_sum = float(track.pop("requested_effort_sum"))
    sampled_producer_sum = float(track.pop("sampled_producer_sum_mol_m3"))
    final = final_by_id.get(int(track["id"]))
    return {
        **track,
        "final_alive": final is not None,
        "final_reserve_q": None if final is None else final["reserve_q"],
        "generation": None if final is None else final["generation"],
        "behavior_intervals": intervals,
        "food_gradient_intervals": food_gradient_intervals,
        "food_gradient_fraction": (
            food_gradient_intervals / intervals if intervals else 0.0
        ),
        "locomoting_intervals": locomoting_intervals,
        "locomoting_fraction": (
            locomoting_intervals / intervals if intervals else 0.0
        ),
        "mean_requested_effort": (requested_effort_sum / intervals if intervals else 0.0),
        "mean_sampled_producer_mol_m3": (sampled_producer_sum / intervals if intervals else 0.0),
        "displacement_m": displacement_m,
        "path_to_displacement_ratio": (path_length_m / displacement_m if displacement_m > 0.0 else None),
    }


def run_foraging_diagnostic(
    *,
    device_name: str,
    duration_s: float = 30.0,
    seed: int = 20260809,
    compile_domains: bool | None = None,
) -> dict[str, object]:
    """Observe per-identity food state, intent, feeding, and trajectories."""

    intervals = _whole_intervals(duration_s)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2^63)")
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    compile_runtime = device.type == "cuda" if compile_domains is None else compile_domains
    world = _build_server_world(device=device)
    _seed_visible_baseline(world, seed=seed)
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=compile_runtime,
        profile=CAUSAL_RUNTIME_PROFILE,
    )
    initial_producer = backend.session.state.economy.bp_q.detach().cpu().clone()
    producer_grazing_by_cell = torch.zeros_like(initial_producer)
    producer_production_q = 0
    producer_maintenance_q = 0
    producer_mortality_q = 0
    tracks: dict[int, dict[str, object]] = {}
    _observe_motion(
        tracks,
        backend.session.state,
        lx_m=world.geometry.lx_m,
        ly_m=world.geometry.ly_m,
    )
    books_closed = True
    with torch.inference_mode():
        for _ in range(intervals):
            before = backend.session.state
            chunk = backend.session.advance_autonomous_chunk(
                backend.fluid,
                intervals=1,
            )
            _observe_behavior(tracks, before, chunk)
            economy_ledger = chunk.last_interval.economy.ledger
            producer_production_q += int(
                economy_ledger.production_q.sum().detach().cpu()
            )
            producer_maintenance_q += int(
                economy_ledger.producer_maintenance_q.sum().detach().cpu()
            )
            producer_mortality_q += int(
                economy_ledger.producer_mortality_q.sum().detach().cpu()
            )
            producer_grazing_by_cell += (
                chunk.last_interval.feeding.ledger.producer_debit_by_cell_q
                .detach()
                .cpu()
            )
            _observe_motion(
                tracks,
                chunk.state,
                lx_m=world.geometry.lx_m,
                ly_m=world.geometry.ly_m,
            )
            books_closed &= bool(
                chunk.last_interval.economy.ledger.books_closed.all().detach().cpu()
            ) and bool(chunk.last_interval.matter.books_closed.all().detach().cpu())

    final = backend.session.state.population
    final_alive = final.alive[0].detach().cpu()
    final_ids = final.stable_id[0].detach().cpu()
    final_reserve = final.reserve_q[0].detach().cpu()
    final_generation = final.generation[0].detach().cpu()
    final_by_id = {
        int(final_ids[slot]): {
            "reserve_q": int(final_reserve[slot]),
            "generation": int(final_generation[slot]),
        }
        for slot in final_alive.nonzero().flatten().tolist()
    }
    creatures = [_finalize_track(track, final_by_id) for _, track in sorted(tracks.items())]
    food_gradient_intervals = sum(
        int(creature["food_gradient_intervals"]) for creature in creatures
    )
    locomoting_intervals = sum(
        int(creature["locomoting_intervals"]) for creature in creatures
    )
    behavior_intervals = sum(int(creature["behavior_intervals"]) for creature in creatures)
    final_producer = backend.session.state.economy.bp_q.detach().cpu()
    initial_producer_q = int(initial_producer.sum())
    final_producer_q = int(final_producer.sum())
    feeding_debit_q = int(producer_grazing_by_cell.sum())
    producer_balance_expected_q = (
        initial_producer_q
        + producer_production_q
        - producer_maintenance_q
        - producer_mortality_q
        - feeding_debit_q
    )
    grazed_cells = producer_grazing_by_cell > 0
    initial_stock_in_grazed_cells_q = int(initial_producer[grazed_cells].sum())
    return {
        "schema": "sirrobin.foraging-diagnostic.v3",
        "interpretation": INTERPRETATION,
        "configuration": {
            "device": str(device),
            "compiled_domains": compile_runtime,
            "seed": seed,
            "duration_s": duration_s,
            "authoritative_interval_s": ECONOMY_INTERVAL_S,
            "intervals": intervals,
            "initial_bodies": INITIAL_BODIES,
        },
        "conservation": {"books_closed": books_closed},
        "producer_accounting": {
            "initial_q": initial_producer_q,
            "final_q": final_producer_q,
            "reaction_production_q": producer_production_q,
            "maintenance_q": producer_maintenance_q,
            "mortality_q": producer_mortality_q,
            "feeding_debit_q": feeding_debit_q,
            "expected_final_q": producer_balance_expected_q,
            "balance_closed": final_producer_q == producer_balance_expected_q,
            "initial_occupied_cells": int((initial_producer > 0).sum()),
            "grazed_cells": int(grazed_cells.sum()),
            "initial_stock_in_grazed_cells_q": initial_stock_in_grazed_cells_q,
            "max_cell_feeding_debit_q": int(producer_grazing_by_cell.max()),
            "feeding_fraction_of_initial_grazed_stock": (
                feeding_debit_q / initial_stock_in_grazed_cells_q
                if initial_stock_in_grazed_cells_q > 0
                else None
            ),
        },
        "aggregate": {
            "observed_identities": len(creatures),
            "behavior_intervals": behavior_intervals,
            "food_gradient_intervals": food_gradient_intervals,
            "locomoting_intervals": locomoting_intervals,
            "feeding_debit_q": feeding_debit_q,
        },
        "creatures": creatures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="disable torch compilation (useful for short CPU diagnostics)",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_foraging_diagnostic(
        device_name=args.device,
        duration_s=args.duration_s,
        seed=args.seed,
        compile_domains=False if args.eager else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
