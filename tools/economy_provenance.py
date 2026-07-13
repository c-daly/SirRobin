"""Hash the exact S1 production/validation source tree used by measured artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def economy_source_hash(root: Path) -> str:
    paths = [
        *(root / "src/sirrobin/economy").glob("*.py"),
        *(root / "src/sirrobin/fields").glob("*.py"),
        root / "src/sirrobin/numerics/flux.py",
        root / "src/sirrobin/observe/economy.py",
        root / "src/sirrobin/validation/economy.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
