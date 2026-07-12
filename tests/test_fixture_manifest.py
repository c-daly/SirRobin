import hashlib
import json
from pathlib import Path

from sirrobin.physics.config import LocomotionConfig


def test_fixture_manifest_hash_closes_every_frozen_input():
    fixtures = Path("oracle/fixtures")
    manifest = json.loads((fixtures / "manifest.json").read_text())
    assert manifest["schema"] == "sirrobin.locomotion.fixtures.v1"
    assert manifest["config_sha256"] == LocomotionConfig().sha256()
    for name, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((fixtures / name).read_bytes()).hexdigest() == expected
    corpus = json.loads((fixtures / "corpus.json").read_text())
    row_order = "\n".join(row["id"] for row in corpus["bodies"]) + "\n"
    assert hashlib.sha256(row_order.encode()).hexdigest() == manifest["row_order_sha256"]
    assert manifest["kg_per_sim_mass"] == 250.0
    assert manifest["sim_length_m"] == 1.0
