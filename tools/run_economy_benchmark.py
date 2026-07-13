"""Benchmark the authorizing S1 grid without changing its scientific configuration."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from economy_provenance import economy_source_hash

from sirrobin.economy.config import DEFAULT_ECONOMY_CONFIG
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rung", choices=("eager", "compile"), default="eager")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DEFAULT_ECONOMY_CONFIG
    state = EconomyState.zeros(config, device=args.device)
    state.nd_q.fill_(500_000_000)
    state.bp_q.fill_(10_000_000)
    state.bd_q[..., 0].fill_(1_000_000)
    kernel = EconomyKernel(state, config)
    step_fn = kernel.step
    if args.rung == "compile":
        step_fn = torch.compile(kernel.step, mode="reduce-overhead")
    for _ in range(args.warmup):
        step_fn()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    closed = torch.ones(config.worlds, dtype=torch.bool, device=args.device)
    interventions = torch.zeros(config.worlds, dtype=torch.int64, device=args.device)
    shortfalls = torch.zeros(config.worlds, dtype=torch.int64, device=args.device)
    samples = []
    for _ in range(args.repetitions):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        for _ in range(args.steps):
            ledger = step_fn()
            closed &= ledger.books_closed
            interventions += ledger.intervention_count
            shortfalls += ledger.transport_shortfall_q
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started
        samples.append(args.steps / elapsed)
    peak = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0
    cells = config.worlds * config.gx * config.gy * config.gz
    payload = {
        "schema": "sirrobin.economy.benchmark.v1",
        "device": args.device,
        "rung": args.rung,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "config_hash": config.sha256(),
        "source_hash": economy_source_hash(Path(__file__).resolve().parents[1]),
        "shape": list(config.shape),
        "steps": args.steps,
        "repetitions": args.repetitions,
        "step_rate_samples": samples,
        "median_steps_per_s": statistics.median(samples),
        "minimum_steps_per_s": min(samples),
        "median_cell_updates_per_s": cells * statistics.median(samples),
        "peak_memory_bytes": int(peak),
        "books_closed": bool(closed.all().item()),
        "intervention_count": int(interventions.sum().item()),
        "transport_shortfall_q": int(shortfalls.sum().item()),
    }
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(args.device)
        payload["hardware"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "cuda_runtime": torch.version.cuda,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not payload["books_closed"] or payload["intervention_count"] or payload["transport_shortfall_q"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
