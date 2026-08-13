#!/usr/bin/env python3
"""Run deterministic replicated lifecycle observations on the living runtime."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace

import torch

from tools.runtime_unity import (
    EVOLUTION_DEMO_RUNTIME_PROFILE,
    RUNTIME_UNITY_PROFILES,
    RuntimeUnityBackend,
    RuntimeUnityProfile,
    runtime_diagnostics,
)
from tools.serve_unity import (
    CAPACITY,
    ECONOMY_INTERVAL_S,
    INITIAL_BODIES,
    LIVE_INITIAL_RESERVE_Q,
    LIVE_RICH_FOOD_CELL_Q,
    _build_server_world,
    _seed_visible_baseline,
)

INTERPRETATION = (
    "observed lifecycle outcomes; extinction and persistence are not pass/fail"
)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _whole_intervals(name: str, duration_s: float, interval_s: float) -> int:
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(duration_s)
        or duration_s <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    ratio = duration_s / interval_s
    intervals = round(ratio)
    if not math.isclose(ratio, intervals, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must contain whole authoritative intervals")
    return intervals


def _seeded_profile(profile: RuntimeUnityProfile, seed: int) -> RuntimeUnityProfile:
    return replace(
        profile,
        mortality=replace(profile.mortality, seed=seed),
        mutation=replace(profile.mutation, seed=seed),
    )


def _sample(backend: RuntimeUnityBackend) -> dict[str, object]:
    snapshot = backend.snapshot()
    diagnostics = runtime_diagnostics(snapshot, backend.config, backend.observation)
    current = diagnostics["current"]
    observed = diagnostics["observed_session"]
    interval = backend.last_interval
    return {
        "step": snapshot.step,
        "time_s": snapshot.time_s,
        "population": current["population"],
        "free_slots": current["free_slots"],
        "age_s": current["age_s"],
        "generation": current["generation"],
        "reserve_q": current["reserve_q"],
        "food_availability": {
            "producer_q": current["producer_q"]["total"],
            "occupied_cells": current["producer_q"]["occupied_cells"],
            "peak_cell_q": current["producer_q"]["peak_cell"],
            "feeding_requested_q": observed["feeding_requested_q"],
            "feeding_actual_debit_q": observed["feeding_actual_debit_q"],
            "feeding_reserve_credit_q": observed["feeding_reserve_credit_q"],
        },
        "reproduction": {
            "clone_funded_parents": current["clone_funded_parents"],
            "requested_births": observed["requested_births"],
            "births": observed["births"],
            "unfunded_birth_rejections": observed["unfunded_birth_rejections"],
            "capacity_birth_rejections": observed["capacity_birth_rejections"],
            "id_birth_rejections": observed["id_birth_rejections"],
        },
        "mortality": {
            "deaths": observed["deaths"],
            "starvation_deaths": observed["starvation_deaths"],
            "old_age_deaths": observed["old_age_deaths"],
        },
        "mutation": {
            "mutated_births": observed["mutated_births"],
            "events": observed["mutation_events"],
            "parameter_events": observed["parameter_mutation_events"],
            "topology_events": observed["topology_mutation_events"],
        },
        "behavior": {
            "seeking_intervals": observed["behavior_seeking_intervals"],
            "searching_intervals": observed["behavior_searching_intervals"],
            "cruising_intervals": observed["behavior_cruising_intervals"],
            "idle_intervals": observed["behavior_idle_intervals"],
        },
        "conservation": {
            "economy_books_closed": (
                None
                if interval is None
                else bool(interval.economy.ledger.books_closed.all())
            ),
            "matter_books_closed": (
                None if interval is None else bool(interval.matter.books_closed.all())
            ),
        },
    }


def run_replicated_lifecycle_diagnostic(
    *,
    device_name: str,
    profile_name: str = EVOLUTION_DEMO_RUNTIME_PROFILE.name,
    replicates: int = 5,
    duration_s: float = 600.0,
    sample_every_s: float = 5.0,
    base_seed: int = 20260813,
    compile_domains: bool | None = None,
) -> dict[str, object]:
    """Observe declared stochastic replicates without assigning a success verdict."""

    _positive_int("replicates", replicates)
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not 0 <= base_seed or base_seed + replicates - 1 >= 2**63:
        raise ValueError("replicate seeds must be in [0,2^63)")
    if compile_domains is not None and not isinstance(compile_domains, bool):
        raise TypeError("compile_domains must be a boolean or None")
    try:
        device = torch.device(device_name)
    except RuntimeError as error:
        raise ValueError(f"invalid device {device_name!r}") from error
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    try:
        base_profile = RUNTIME_UNITY_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(f"unknown runtime profile {profile_name!r}") from error

    interval_s = ECONOMY_INTERVAL_S
    requested_intervals = _whole_intervals("duration_s", duration_s, interval_s)
    sample_intervals = _whole_intervals("sample_every_s", sample_every_s, interval_s)
    compile_runtime = (
        device.type == "cuda" if compile_domains is None else compile_domains
    )
    replicate_seeds = [base_seed + index for index in range(replicates)]
    reports: list[dict[str, object]] = []

    with torch.inference_mode():
        for replicate, seed in enumerate(replicate_seeds):
            profile = _seeded_profile(base_profile, seed)
            world = _build_server_world(device=device)
            _seed_visible_baseline(world, seed=seed)
            backend = RuntimeUnityBackend.from_reference_fixture(
                world,
                compile_domains=compile_runtime,
                profile=profile,
            )
            samples = [_sample(backend)]
            terminal_reason = "horizon"
            for interval_index in range(1, requested_intervals + 1):
                events = backend.advance_events()
                extinct = not bool(events.alive.any())
                if (
                    interval_index % sample_intervals == 0
                    or interval_index == requested_intervals
                    or extinct
                ):
                    samples.append(_sample(backend))
                if extinct:
                    terminal_reason = "extinction"
                    break
            reports.append(
                {
                    "replicate": replicate,
                    "seed": seed,
                    "causal_inputs": {
                        "position_seed": seed,
                        "mortality_seed": profile.mortality.seed,
                        "mutation_seed": profile.mutation.seed,
                        "initial_population": samples[0]["population"],
                        "initial_reserve_q_per_creature": LIVE_INITIAL_RESERVE_Q,
                        "rich_food_cell_q": LIVE_RICH_FOOD_CELL_Q,
                    },
                    "terminal_reason": terminal_reason,
                    "samples": samples,
                }
            )

    terminal_counts = Counter(report["terminal_reason"] for report in reports)
    final_samples = [report["samples"][-1] for report in reports]
    final_populations = [sample["population"] for sample in final_samples]
    return {
        "schema": "sirrobin.lifecycle-replicates.v1",
        "interpretation": INTERPRETATION,
        "configuration": {
            "device": str(device),
            "compiled_domains": compile_runtime,
            "profile": base_profile.name,
            "profile_description": base_profile.description,
            "replicates": replicates,
            "base_seed": base_seed,
            "replicate_seeds": replicate_seeds,
            "duration_s": duration_s,
            "sample_every_s": sample_every_s,
            "authoritative_interval_s": interval_s,
            "requested_intervals": requested_intervals,
            "capacity": CAPACITY,
            "initial_bodies": INITIAL_BODIES,
            "mortality": {
                "min_lifespan_s": base_profile.mortality.min_lifespan_s,
                "max_lifespan_s": base_profile.mortality.max_lifespan_s,
            },
            "mutation": {
                "rate_per_locus": base_profile.mutation.mutation_rate_per_locus,
                "max_events_per_birth": base_profile.mutation.max_mutations_per_birth,
                "parameter_event_weight": base_profile.mutation.parameter_event_weight,
                "topology_event_weight": base_profile.mutation.topology_event_weight,
            },
        },
        "replicates": reports,
        "aggregate": {
            "terminal_reason_counts": dict(sorted(terminal_counts.items())),
            "final_population": {
                "min": min(final_populations),
                "mean": math.fsum(final_populations) / len(final_populations),
                "max": max(final_populations),
            },
            "births": sum(sample["reproduction"]["births"] for sample in final_samples),
            "deaths": sum(sample["mortality"]["deaths"] for sample in final_samples),
            "mutation_events": sum(
                sample["mutation"]["events"] for sample in final_samples
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--profile",
        choices=tuple(RUNTIME_UNITY_PROFILES),
        default=EVOLUTION_DEMO_RUNTIME_PROFILE.name,
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--sample-every-s", type=float, default=5.0)
    parser.add_argument("--base-seed", type=int, default=20260813)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="disable torch compilation (useful for short CPU diagnostics)",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_replicated_lifecycle_diagnostic(
        device_name=args.device,
        profile_name=args.profile,
        replicates=args.replicates,
        duration_s=args.duration_s,
        sample_every_s=args.sample_every_s,
        base_seed=args.base_seed,
        compile_domains=False if args.eager else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
