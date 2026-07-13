"""S2 heterogeneous live-locomotion benchmark harness."""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from sirrobin.core.live_world import advance_live_world, initialize_live_state
from sirrobin.fields.geometry import GridGeometry
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig


@dataclass(frozen=True, slots=True)
class LiveBenchmarkResult:
    corpus_class: str
    capacity: int
    live_bodies: int
    device: str
    rung: str
    steps: int
    repetitions: tuple[float, ...]
    median_creature_steps_s: float
    minimum_creature_steps_s: float
    q1_creature_steps_s: float
    q3_creature_steps_s: float
    peak_memory_bytes: int
    compile_time_s: float
    regularization_count: int
    inertia_floor_count: int
    omega_backstop_count: int
    nonfinite_count: int
    status: str


def _replicated_genotype(
    donor: dict[str, Any], corpus: dict[str, Any], class_name: str, capacity: int, live: int
) -> GenotypeBatch:
    by_id = {row["id"]: row for row in donor["bodies"]}
    cycle = corpus["classes"][class_name]["cycle_ids"]
    base = GenotypeBatch.from_donor_rows([by_id[body_id] for body_id in cycle], dtype=torch.float32)
    index = torch.arange(capacity) % len(cycle)
    values: dict[str, torch.Tensor] = {}
    for field in fields(base):
        values[field.name] = getattr(base, field.name)[0, index].unsqueeze(0).clone()
    values["stable_id"] = torch.arange(1, capacity + 1, dtype=torch.int64).unsqueeze(0)
    values["alive"] = (torch.arange(capacity) < live).unsqueeze(0)
    result = GenotypeBatch(**values)
    result.validate()
    return result


def benchmark_live_cell(
    donor: dict[str, Any],
    corpus: dict[str, Any],
    class_name: str,
    *,
    capacity: int,
    live: int,
    device: str,
    rung: str = "eager",
    steps: int = 120,
    warmup: int = 10,
    repetitions: int = 5,
) -> LiveBenchmarkResult:
    if rung not in {"eager", "compile", "cudagraph"}:
        raise ValueError("live benchmark rung must be eager, compile, or cudagraph")
    if rung == "cudagraph" and not device.startswith("cuda"):
        raise ValueError("cudagraph rung requires CUDA")
    compile_time = 0.0
    try:
        genotype = _replicated_genotype(donor, corpus, class_name, capacity, live).to(device)
        body = develop(genotype)
        state = initialize_live_state(body)
        state.position_enu_m[..., 2] = -10.0
        config = LiveLocomotionConfig()
        fluid = FluidSample(
            torch.full(body.alive.shape, config.rho_water, dtype=torch.float32, device=device),
            torch.zeros((*body.alive.shape, 3), dtype=torch.float32, device=device),
        )
        geometry = GridGeometry(32, 32, 8, 1000.0, 1000.0, 100.0)
        requested = torch.zeros((*body.alive.shape, 2), dtype=torch.float32, device=device)
        requested[..., 0] = 1.0
        counters = torch.zeros(4, dtype=torch.int64, device=device)

        def step_once():
            ledger = advance_live_world(
                body,
                state,
                fluid,
                config,
                geometry,
                requested_heading_enu=requested,
            )
            counters[0] += ledger.solve_regularized.sum()
            counters[1] += ledger.yaw_inertia_floor_hit.sum()
            counters[2] += ledger.omega_backstop_hit.sum()
            counters[3] += ledger.nonfinite.sum()
            return ledger

        step_fn = step_once
        if rung == "compile":
            started = time.perf_counter()
            step_fn = torch.compile(step_once, mode="reduce-overhead")
            step_fn()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            compile_time = time.perf_counter() - started
        elif rung == "cudagraph":
            side_stream = torch.cuda.Stream(device=device)
            side_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    step_once()
            torch.cuda.current_stream(device).wait_stream(side_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_ledger = step_once()

            def replay_graph():
                graph.replay()
                return static_ledger

            step_fn = replay_graph
        for _ in range(warmup):
            step_fn()
        counters.zero_()
        samples = []
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repetitions):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(steps):
                step_fn()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            samples.append(live * steps / elapsed)
        ordered = sorted(samples)
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        counts = [int(value) for value in counters.cpu().tolist()]
        peak = torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0
        return LiveBenchmarkResult(
            class_name,
            capacity,
            live,
            device,
            rung,
            steps,
            tuple(samples),
            statistics.median(samples),
            min(samples),
            quartiles[0],
            quartiles[2],
            int(peak),
            compile_time,
            *counts,
            "ok" if not any(counts) else "intervention",
        )
    except torch.OutOfMemoryError:
        return LiveBenchmarkResult(
            class_name,
            capacity,
            live,
            device,
            rung,
            steps,
            (),
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            compile_time,
            0,
            0,
            0,
            0,
            "oom",
        )


def write_live_benchmark(
    path: Path,
    result: LiveBenchmarkResult,
    config: LiveLocomotionConfig,
    fixture_hashes: dict[str, str],
) -> None:
    payload: dict[str, Any] = {
        "schema": "sirrobin.live.benchmark.v1",
        "result": asdict(result),
        "config_hash": config.sha256(),
        "fixture_hashes": fixture_hashes,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if result.device.startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(result.device)
        payload["cuda"] = {
            "runtime": torch.version.cuda,
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
