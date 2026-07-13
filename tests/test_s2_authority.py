from dataclasses import fields
from pathlib import Path

from sirrobin.physics.contracts import LiveState

ROOT = Path(__file__).resolve().parents[1]


def test_active_docs_route_s2_to_recorded_no_go():
    required = {
        "CLAUDE.md": "S2 is implemented but records NO-GO",
        "docs/2026-07-11-sirrobin-design-document.md": "S2 is implemented but records NO-GO",
        "docs/2026-07-12-sirrobin-developer-reference.md": "S2 is implemented but records NO-GO",
    }
    for relative, phrase in required.items():
        text = " ".join((ROOT / relative).read_text(encoding="utf-8").replace(">", "").split())
        assert phrase in text
    assert (ROOT / "docs/superpowers/reports/2026-07-12-sirrobin-S2-decision-report.md").is_file()


def test_live_state_has_no_cached_capability_or_duplicate_yaw_rate():
    names = {field.name for field in fields(LiveState)}
    forbidden = {"omega", "yaw_rate", "speed", "eff", "swim_proxy", "defense_score", "agility_score"}
    assert names.isdisjoint(forbidden)
