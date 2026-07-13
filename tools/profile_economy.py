"""Profile the finalized full-grid economy step and emit top operator attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from economy_provenance import economy_source_hash

from sirrobin.economy.config import DEFAULT_ECONOMY_CONFIG
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/economy-profile.json"))
    args = parser.parse_args()
    config = DEFAULT_ECONOMY_CONFIG
    state = EconomyState.zeros(config, device=args.device)
    state.nd_q.fill_(500_000_000)
    state.bp_q.fill_(10_000_000)
    state.bd_q[..., 0].fill_(1_000_000)
    kernel = EconomyKernel(state, config)
    for _ in range(2):
        kernel.step()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=True) as profile:
        for _ in range(args.steps):
            kernel.step()
    rows = []
    def self_time(event: object) -> float:
        if args.device.startswith("cuda"):
            return float(getattr(event, "self_device_time_total", 0.0))
        return float(event.self_cpu_time_total)

    for event in sorted(profile.key_averages(), key=self_time, reverse=True)[:15]:
        rows.append(
            {
                "operator": event.key,
                "calls": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
                "self_cuda_time_us": getattr(event, "self_device_time_total", 0.0),
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "self_cuda_memory_bytes": getattr(event, "self_device_memory_usage", 0),
            }
        )
    payload = {
        "schema": "sirrobin.economy.profile.v1",
        "device": args.device,
        "steps": args.steps,
        "config_hash": config.sha256(),
        "source_hash": economy_source_hash(Path(__file__).resolve().parents[1]),
        "top_operators": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
