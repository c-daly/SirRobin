"""Run one fresh-process benchmark cell and write its provenance manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sirrobin.benchmarks.locomotion import benchmark_cell, write_result  # noqa: E402
from sirrobin.physics.config import LocomotionConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class", dest="class_name", choices=("H0", "H1", "H2", "FULL"), required=True
    )
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--live", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--rung", choices=("r0-eager", "r1-compile", "r2-cudagraph"), required=True
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=360)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--churn", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus_path = ROOT / "oracle" / "fixtures" / "corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_hash = (ROOT / "oracle" / "fixtures" / "corpus.sha256").read_text().split()[0]
    config = LocomotionConfig(n_cap=args.capacity, n_live=args.live)
    result = benchmark_cell(
        corpus,
        args.class_name,
        capacity=args.capacity,
        live=args.live,
        device=args.device,
        rung=args.rung,
        steps=args.steps,
        warmup=args.warmup,
        repetitions=args.repetitions,
        churn=args.churn,
    )
    write_result(args.output, result, config, corpus_hash)
    print(json.dumps({"status": result.status, "minimum": result.minimum, "median": result.median}))


if __name__ == "__main__":
    main()
