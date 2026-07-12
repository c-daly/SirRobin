"""Report same-device eager float replay divergence and deterministic-mode tax."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sirrobin.physics.config import LocomotionConfig  # noqa: E402
from sirrobin.physics.contracts import BodyBatch  # noqa: E402
from sirrobin.physics.swim_step import SwimKernel  # noqa: E402


def execute(rows: list[dict], config: LocomotionConfig, *, deterministic: bool, steps: int):
    torch.use_deterministic_algorithms(deterministic)
    body = BodyBatch.from_rows(rows, config, dtype=torch.float32)
    kernel = SwimKernel(body, config)
    trace = torch.empty((steps, len(rows), 7), dtype=torch.float32)
    started = time.perf_counter()
    for index in range(steps):
        ledger = kernel.step()
        trace[index, :, :3] = body.v_com
        trace[index, :, 3:6] = body.x_com
        trace[index, :, 6] = ledger.r_step
    elapsed = time.perf_counter() - started
    return trace, elapsed


def main() -> None:
    corpus_path = ROOT / "oracle" / "fixtures" / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    by_id = {row["id"]: row for row in corpus["bodies"]}
    ids = ("H1-00", "H1-31", "H2-03", "H2-58")
    rows = [by_id[body_id] for body_id in ids]
    config = LocomotionConfig(n_cap=len(rows), n_live=len(rows))
    normal, normal_seconds = execute(rows, config, deterministic=False, steps=960)
    first, first_seconds = execute(rows, config, deterministic=True, steps=960)
    second, second_seconds = execute(rows, config, deterministic=True, steps=960)
    output = {
        "schema": "sirrobin.locomotion.determinism-diagnostic.v1",
        "device": "cpu",
        "dtype": "float32",
        "steps": 960,
        "body_ids": ids,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "deterministic_rerun_max_abs": float((first - second).abs().max()),
        "normal_vs_deterministic_max_abs": float((normal - first).abs().max()),
        "normal_seconds": normal_seconds,
        "deterministic_seconds": [first_seconds, second_seconds],
        "deterministic_tax": ((first_seconds + second_seconds) * 0.5) / normal_seconds,
        "torch_version": torch.__version__,
    }
    path = ROOT / "runs" / "determinism-cpu.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
