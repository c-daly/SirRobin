from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_reservoir_mutation_is_confined_to_transaction_modules() -> None:
    allowed = {
        ROOT / "src/sirrobin/economy/reactions.py",
        ROOT / "src/sirrobin/fields/transport.py",
        ROOT / "src/sirrobin/fields/grid.py",
        ROOT / "src/sirrobin/validation/economy.py",
    }
    reservoir_names = {"nd_q", "bp_q", "bd_q", "bm_q"}
    mutators = {"add_", "sub_", "copy_", "fill_", "zero_"}
    violations: list[str] = []
    for path in (ROOT / "src/sirrobin").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                node.func.attr in mutators
                and isinstance(owner, ast.Attribute)
                and owner.attr in reservoir_names
            ):
                if path not in allowed:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "raw reservoir mutation outside transactions: " + ", ".join(violations)
