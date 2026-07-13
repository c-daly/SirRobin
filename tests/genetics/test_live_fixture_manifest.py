import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "oracle/fixtures/live"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_live_fixture_manifest_hashes_are_frozen():
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["fixtures"].items():
        assert _sha(FIXTURES / relative) == expected
    for relative, expected in manifest["generators"].items():
        assert _sha(ROOT / relative) == expected


def test_live_oracle_is_independent_and_fixtures_are_nontrivial():
    source = (ROOT / "tools/live_oracle.py").read_text(encoding="utf-8")
    assert "import sirrobin" not in source
    donor = json.loads((FIXTURES / "donor_development_live.json").read_text(encoding="utf-8"))
    gain1 = json.loads((FIXTURES / "gain1_canonical.json").read_text(encoding="utf-8"))
    assert len(donor["bodies"]) == len(gain1["bodies"]) == 32
    assert any(len(body["segments"]) == 16 for body in donor["bodies"])
    assert all(body["tail_aft_dot"] < 0 for body in gain1["bodies"])
    assert max(abs(case["residual"]) for case in gain1["yaw_cases"]["angular_work"]) < 1e-14


def test_live_corpus_pins_population_authority():
    corpus = json.loads((FIXTURES / "corpus.json").read_text(encoding="utf-8"))
    assert set(corpus["classes"]) == {"H1", "H2"}
    assert corpus["authorization"]["h0_may_authorize"] is False
    assert corpus["authorization"]["population_cells"] == [
        {"live": 5000, "capacity": 5120, "floor_creature_steps_s": 600000},
        {"live": 10000, "capacity": 10240, "floor_creature_steps_s": 1200000},
    ]
    h2 = corpus["classes"]["H2"]["cycle_ids"]
    assert h2.count("wide-16") == 3
    assert len(h2) == 20
    donor = json.loads((FIXTURES / "donor_development_live.json").read_text(encoding="utf-8"))
    counts = {body["id"]: len(body["segments"]) for body in donor["bodies"]}
    assert min(counts[body_id] for body_id in h2) >= 2
