"""Public read-only S0 contracts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: Path
    sha256: str
