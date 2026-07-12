"""Collect operator attribution for one locomotion corpus/device cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sirrobin.benchmarks.locomotion import replicate_rows  # noqa: E402
from sirrobin.physics.config import LocomotionConfig  # noqa: E402
from sirrobin.physics.contracts import BodyBatch  # noqa: E402
from sirrobin.physics.swim_step import SwimKernel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name", choices=("H0", "H1", "H2"), required=True)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--live", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = json.loads((ROOT / "oracle" / "fixtures" / "corpus.json").read_text())
    rows, _ = replicate_rows(corpus, args.class_name, args.capacity, args.live)
    config = LocomotionConfig(n_cap=args.capacity, n_live=args.live)
    body = BodyBatch.from_rows(rows, config, dtype=torch.float32, device=args.device)
    body.alive[args.live :] = False
    kernel = SwimKernel(body, config)
    for _ in range(10):
        kernel.step()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        for _ in range(args.steps):
            kernel.step()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)

    rows_out = []
    for event in profile.key_averages():
        device_us = float(getattr(event, "self_device_time_total", 0.0))
        rows_out.append(
            {
                "operator": event.key,
                "calls": event.count,
                "self_cpu_us": float(event.self_cpu_time_total),
                "self_device_us": device_us,
            }
        )
    rows_out.sort(key=lambda row: (row["self_device_us"], row["self_cpu_us"]), reverse=True)
    payload = {
        "schema": "sirrobin.locomotion.profile.v1",
        "corpus_class": args.class_name,
        "capacity": args.capacity,
        "live": args.live,
        "device": args.device,
        "steps": args.steps,
        "torch_version": torch.__version__,
        "corpus_sha256": hashlib.sha256(
            (ROOT / "oracle" / "fixtures" / "corpus.json").read_bytes()
        ).hexdigest(),
        "class_rows_sha256": hashlib.sha256(
            json.dumps(
                [row for row in corpus["bodies"] if row["class"] == args.class_name],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "top_operators": rows_out[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
