"""Measure the complete candidate living interval without rendering."""

from __future__ import annotations

import argparse
import json
import time

import torch

from sirrobin.organisms.behavior import BehaviorConfig
from sirrobin.organisms.development import calibrate_development_config
from sirrobin.organisms.feeding import FeedingConfig
from sirrobin.organisms.metabolism import MetabolismConfig
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.organisms.mutation import MutationConfig
from sirrobin.physics.phase_response import PhaseWindowConfig
from sirrobin.runtime.config import LivingRuntimeConfig
from sirrobin.runtime.reference_adapter import living_state_from_reference
from sirrobin.runtime.session import RuntimeSession
from sirrobin.runtime.state import LivingState
from sirrobin.runtime.step import LivingIntervalInputs, advance_living_interval
from tools.run_world import LIVING_MATERIAL_ENERGY_CONFIG, _build_fixture_world


def build_fixture(
    slots: int,
    device: torch.device,
    *,
    allocation_rounds: int = 8,
) -> tuple[LivingState, LivingIntervalInputs, LivingRuntimeConfig]:
    world = _build_fixture_world(
        bodies=slots,
        live_bodies=slots,
        reserve_q_per_creature=5_000,
        device=device,
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    alive = world.genotype.alive
    state = living_state_from_reference(world)
    config = LivingRuntimeConfig(
        economy=world.economy_config,
        live=world.live_config,
        motion=PhaseWindowConfig(0.1, stages=4, phase_samples=2),
        behavior=BehaviorConfig(1.0),
        feeding=FeedingConfig(
            interval_s=0.1,
            q_mass_mol=world.economy_config.q_mass_mol,
            capture_efficiency=0.5,
            assimilation_efficiency=0.5,
            producer_j_per_q=LIVING_MATERIAL_ENERGY_CONFIG.producer_j_per_q,
            reserve_j_per_q=LIVING_MATERIAL_ENERGY_CONFIG.reserve_j_per_q,
            allocation_rounds=allocation_rounds,
        ),
        metabolism=MetabolismConfig(
            interval_s=0.1,
            maintenance_w_per_kg=0.01,
            chemical_to_mechanical_efficiency=1.0,
            reserve_j_per_q=LIVING_MATERIAL_ENERGY_CONFIG.reserve_j_per_q,
        ),
        mortality=MortalityConfig(1.0e9, 1.0e9, seed=7),
        mutation=MutationConfig(seed=11),
        development=calibrate_development_config(state.population, state.body),
        child_initial_reserve_q=100,
    )
    inputs = LivingIntervalInputs(
        fluid=world.fluid,
        requested_effort=torch.ones_like(alive, dtype=torch.float32),
        birth_requested=torch.zeros_like(alive),
    )
    return state, inputs, config


def benchmark(
    *,
    slots: int,
    device: torch.device,
    warmup: int,
    intervals: int,
    compiled_motion: bool,
    compiled_domains: bool,
    allocation_rounds: int,
    optimistic_candidates: bool = True,
) -> dict[str, int | float | str | bool]:
    state, inputs, config = build_fixture(
        slots,
        device,
        allocation_rounds=allocation_rounds,
    )
    session = (
        RuntimeSession(
            state,
            config,
            compile_motion=compiled_motion or compiled_domains,
            compile_domains=compiled_domains,
            optimistic_candidates=optimistic_candidates,
        )
        if compiled_motion or compiled_domains
        else None
    )
    for _ in range(warmup):
        if session is None:
            state = advance_living_interval(state, inputs, config).state
        else:
            state = session.advance_chunk(inputs, intervals=1).state
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if session is None:
        books_closed = torch.ones(
            config.economy.worlds, dtype=torch.bool, device=device
        )
        invalid = torch.zeros_like(books_closed)
        for _ in range(intervals):
            advance = advance_living_interval(state, inputs, config)
            state = advance.state
            books_closed &= advance.ledger.matter.books_closed
            invalid |= advance.ledger.invalid
    else:
        chunk = session.advance_chunk(inputs, intervals=intervals)
        state = chunk.state
        books_closed = chunk.last_interval.matter.books_closed
        invalid = chunk.invalid
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started
    simulated_s = intervals * config.economy.dt_eco_s
    return {
        "device": str(device),
        "compiled_motion": compiled_motion or compiled_domains,
        "compiled_domains": compiled_domains,
        "allocation_rounds": allocation_rounds,
        "optimistic_candidates": optimistic_candidates,
        "slots": slots,
        "intervals": intervals,
        "wall_s": elapsed_s,
        "simulated_s": simulated_s,
        "sim_s_per_wall_s": simulated_s / elapsed_s,
        "creature_intervals_per_s": slots * intervals / elapsed_s,
        "books_closed": bool(books_closed.all().cpu()),
        "invalid": bool(invalid.any().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compiled-motion", action="store_true")
    parser.add_argument("--compiled-domains", action="store_true")
    parser.add_argument("--allocation-rounds", type=int, default=8)
    parser.add_argument("--dense-candidates", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.profile:
        state, inputs, config = build_fixture(
            args.slots,
            device,
            allocation_rounds=args.allocation_rounds,
        )
        session = (
            RuntimeSession(
                state,
                config,
                compile_motion=args.compiled_motion or args.compiled_domains,
                compile_domains=args.compiled_domains,
                optimistic_candidates=not args.dense_candidates,
            )
            if args.compiled_motion or args.compiled_domains
            else None
        )
        if session is None:
            state = advance_living_interval(state, inputs, config).state
        else:
            state = session.advance_chunk(inputs, intervals=1).state
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        ) as profile:
            if session is None:
                advance_living_interval(state, inputs, config)
            else:
                session.advance_chunk(inputs, intervals=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        sort_key = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        print(profile.key_averages().table(sort_by=sort_key, row_limit=30))
    else:
        print(
            json.dumps(
                benchmark(
                    slots=args.slots,
                    device=device,
                    warmup=args.warmup,
                    intervals=args.intervals,
                    compiled_motion=args.compiled_motion,
                    compiled_domains=args.compiled_domains,
                    allocation_rounds=args.allocation_rounds,
                    optimistic_candidates=not args.dense_candidates,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
