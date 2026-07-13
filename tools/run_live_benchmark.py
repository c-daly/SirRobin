#!/usr/bin/env python3
"""Run one source-bound S2 live benchmark cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from sirrobin.benchmarks.live import benchmark_live_cell, write_live_benchmark
from sirrobin.physics.live_config import LiveLocomotionConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "oracle/fixtures/live"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name", choices=("H1", "H2"), required=True)
    parser.add_argument("--live", type=int, required=True)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rung", choices=("eager", "compile", "cudagraph"), default="eager")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capacity = args.capacity or (5120 if args.live == 5000 else 10240)
    if args.live <= 0 or args.live > capacity:
        parser.error("--live must lie in [1, capacity]")
    donor_path = FIXTURE_DIR / "donor_development_live.json"
    corpus_path = FIXTURE_DIR / "corpus.json"
    donor = json.loads(donor_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    result = benchmark_live_cell(
        donor,
        corpus,
        args.class_name,
        capacity=capacity,
        live=args.live,
        device=args.device,
        rung=args.rung,
        steps=args.steps,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    write_live_benchmark(
        args.output,
        result,
        LiveLocomotionConfig(),
        {"donor": _sha(donor_path), "corpus": _sha(corpus_path)},
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
