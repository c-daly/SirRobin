from dataclasses import fields
from pathlib import Path

from sirrobin.physics.contracts import LiveState

ROOT = Path(__file__).resolve().parents[1]


def test_active_docs_route_to_living_loop_recovery_authority():
    authority_name = "2026-07-13-sirrobin-living-loop-recovery-implementation-plan.md"
    authority = ROOT / "docs" / "superpowers" / "plans" / authority_name
    assert authority.is_file()
    for relative in (
        "CLAUDE.md",
        "docs/2026-07-11-sirrobin-design-document.md",
        "docs/2026-07-12-sirrobin-developer-reference.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert authority_name in text

    # The authority changed; the frozen S2 result remains historical evidence.
    assert (ROOT / "docs/superpowers/reports/2026-07-12-sirrobin-S2-decision-report.md").is_file()


def test_live_state_has_no_cached_capability_or_duplicate_yaw_rate():
    names = {field.name for field in fields(LiveState)}
    forbidden = {"omega", "yaw_rate", "speed", "eff", "swim_proxy", "defense_score", "agility_score"}
    assert names.isdisjoint(forbidden)
