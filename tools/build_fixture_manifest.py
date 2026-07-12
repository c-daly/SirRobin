"""Build the hash-closed manifest for all frozen locomotion evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sirrobin.physics.config import LocomotionConfig  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fixtures = ROOT / "oracle" / "fixtures"
    corpus_path = fixtures / "corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    donor = json.loads((fixtures / "gain0_donor.json").read_text(encoding="utf-8"))
    row_order = "\n".join(row["id"] for row in corpus["bodies"]) + "\n"
    artifact_names = (
        "corpus.json",
        "gain0_donor.json",
        "gain1_analytic.json",
        "quadrature_gl256.json",
        "quadrature_gl32_negative.json",
    )
    payload = {
        "schema": "sirrobin.locomotion.fixtures.v1",
        "config_sha256": LocomotionConfig().sha256(),
        "corpus_sha256": sha(corpus_path),
        "row_order_sha256": hashlib.sha256(row_order.encode()).hexdigest(),
        "body_count": len(corpus["bodies"]),
        "artifact_sha256": {name: sha(fixtures / name) for name in artifact_names},
        "generator_sha256": {
            "corpus": sha(ROOT / "tools" / "build_corpus.py"),
            "gain0": sha(ROOT / "oracle" / "Program.cs"),
            "gain1": sha(ROOT / "tools" / "gain1_oracle.py"),
        },
        "donor_path": donor["donor_path"],
        "donor_sha256": donor["donor_sha256"],
        "kg_per_sim_mass": 250.0,
        "sim_length_m": 1.0,
    }
    (fixtures / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
