"""Staged locomotion-kernel benchmark harness."""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sirrobin.benchmarks.lifecycle import apply_fixed_churn
from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch
from sirrobin.physics.swim_step import SwimKernel


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    corpus_class: str
    batch_capacity: int
    live_bodies: int
    device: str
    rung: str
    dtype: str
    steps: int
    repetitions: tuple[float, ...]
    median: float
    minimum: float
    q1: float
    q3: float
    peak_memory_bytes: int
    regularization_count: int
    status: str


def replicate_rows(
    corpus: dict[str, Any], class_name: str, capacity: int, live: int
) -> tuple[list[dict[str, Any]], int]:
    source = [
        row
        for row in corpus["bodies"]
        if row["class"] == class_name
        or (class_name == "FULL" and row["class"] == "H2" and row["segment_count"] == 16)
    ]
    if not source:
        raise ValueError(f"no corpus rows for benchmark class {class_name}")
    rows = [source[i % len(source)] for i in range(capacity)]
    return rows, live


def benchmark_cell(
    corpus: dict[str, Any],
    class_name: str,
    *,
    capacity: int,
    live: int,
    device: str,
    rung: str = "r0-eager",
    steps: int = 600,
    warmup: int = 20,
    repetitions: int = 5,
    churn: bool = False,
) -> BenchmarkResult:
    if rung not in {"r0-eager", "r1-compile", "r2-cudagraph"}:
        raise ValueError(f"unknown benchmark rung: {rung}")
    if rung == "r2-cudagraph" and not device.startswith("cuda"):
        raise ValueError("r2-cudagraph requires a CUDA device")
    dtype = torch.float32
    try:
        rows, live = replicate_rows(corpus, class_name, capacity, live)
        body = BodyBatch.from_rows(
            rows, LocomotionConfig(n_cap=capacity, n_live=live), dtype=dtype, device=device
        )
        body.alive[live:] = False
        kernel = SwimKernel(body, LocomotionConfig(n_cap=capacity, n_live=live))
        regularization_count_device = torch.zeros((), dtype=torch.int64, device=device)

        def step_and_count() -> Any:
            ledger = kernel.step()
            regularization_count_device.add_(ledger.regularized.sum())
            return ledger

        step_fn = step_and_count
        if rung == "r1-compile":
            step_fn = torch.compile(step_and_count, mode="reduce-overhead")
        elif rung == "r2-cudagraph":
            side_stream = torch.cuda.Stream(device=device)
            side_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    step_and_count()
            torch.cuda.current_stream(device).wait_stream(side_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_ledger = step_and_count()

            def replay_graph() -> Any:
                graph.replay()
                return static_ledger

            step_fn = replay_graph
        for _ in range(warmup):
            step_fn()
        samples = []
        regularization_count_device.zero_()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for rep in range(repetitions):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for step in range(steps):
                step_fn()
                if churn:
                    apply_fixed_churn(body, (rep * steps + step + 1), period=1000)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            samples.append(live * steps / elapsed)
        ordered = sorted(samples)
        quartiles = (
            statistics.quantiles(ordered, n=4, method="inclusive")
            if len(ordered) > 1
            else [ordered[0], ordered[0], ordered[0]]
        )
        peak = torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0
        regularization_count = int(regularization_count_device.item())
        return BenchmarkResult(
            class_name,
            capacity,
            live,
            device,
            rung,
            str(dtype).removeprefix("torch."),
            steps,
            tuple(samples),
            statistics.median(samples),
            min(samples),
            quartiles[0],
            quartiles[2],
            int(peak),
            regularization_count,
            "ok",
        )
    except torch.OutOfMemoryError:
        return BenchmarkResult(
            class_name,
            capacity,
            live,
            device,
            rung,
            "float32",
            steps,
            (),
            0,
            0,
            0,
            0,
            0,
            0,
            "oom",
        )


def write_result(path: Path, result: BenchmarkResult, config: LocomotionConfig, corpus_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sirrobin.locomotion.benchmark.v1",
        "result": asdict(result),
        "config_hash": config.sha256(),
        "corpus_hash": corpus_hash,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if result.device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(result.device)
        payload["cuda"] = {
            "runtime": torch.version.cuda,
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
