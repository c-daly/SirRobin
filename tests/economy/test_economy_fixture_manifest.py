from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "oracle/fixtures/economy"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_economy_fixtures_are_hash_closed_and_generator_is_independent() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator_sha256"] == sha256(ROOT / "tools/economy_oracle.py")
    for name, expected in manifest["files"].items():
        assert sha256(FIXTURES / name) == expected
    source = (ROOT / "tools/economy_oracle.py").read_text(encoding="utf-8")
    imports = {
        node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    assert "torch" not in imports
    assert not any(name.startswith("sirrobin") for name in imports)
