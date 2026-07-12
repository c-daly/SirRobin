"""Run and persist the full long-horizon locomotion prefix-budget curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sirrobin.physics.config import LocomotionConfig  # noqa: E402
from sirrobin.physics.contracts import BodyBatch  # noqa: E402
from sirrobin.physics.swim_step import SwimKernel  # noqa: E402
from sirrobin.validation.drift import run_prefix_budget  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus_path = ROOT / "oracle" / "fixtures" / "corpus.json"
    corpus_bytes = corpus_path.read_bytes()
    corpus = json.loads(corpus_bytes)
    ids = ("H1-00", "H1-32", "H2-00", "H2-58")
    by_id = {row["id"]: row for row in corpus["bodies"]}
    rows = [by_id[body_id] for body_id in ids]
    dtype = getattr(torch, args.dtype)
    config = LocomotionConfig(n_cap=len(rows), n_live=len(rows))
    body = BodyBatch.from_rows(rows, config, dtype=dtype)
    result = run_prefix_budget(SwimKernel(body, config), steps=args.steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    curves_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        curves_path,
        cumulative_residual=result.cumulative_residual.cpu().numpy(),
        cumulative_scale=result.cumulative_scale.cpu().numpy(),
        prefix_ratio=result.prefix_ratio.cpu().numpy(),
    )
    curves_hash = hashlib.sha256(curves_path.read_bytes()).hexdigest()
    threshold = 1e-6 if dtype == torch.float64 else 1e-3
    payload = {
        "schema": "sirrobin.locomotion.drift.v1",
        "body_ids": ids,
        "dtype": args.dtype,
        "steps": args.steps,
        "burnin": 100,
        "maximum_after_burnin": result.maximum_after_burnin.cpu().tolist(),
        "monotone_bias": result.monotone_bias.cpu().tolist(),
        "threshold": threshold,
        "passed": bool(torch.all(result.maximum_after_burnin < threshold)),
        "regularization_count": result.regularization_count,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "config_sha256": config.sha256(),
        "curves_file": curves_path.name,
        "curves_sha256": curves_hash,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
