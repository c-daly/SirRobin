"""Load and validate the frozen H0/H1/H2 developed-body corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

EXPECTED_HISTOGRAMS = {
    "H0": {6: 64},
    "H1": {2: 6, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 6, 10: 4, 12: 4, 16: 4},
    "H2": {2: 28, 3: 28, 16: 8},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    validate_corpus(raw)
    return raw


def validate_corpus(corpus: dict[str, Any]) -> None:
    if corpus.get("schema") != "sirrobin.locomotion.corpus.v1":
        raise ValueError("unexpected corpus schema")
    bodies = corpus.get("bodies")
    if not isinstance(bodies, list) or len(bodies) != 192:
        raise ValueError("corpus must contain exactly 192 bodies")
    ids = [body["id"] for body in bodies]
    if len(set(ids)) != len(ids):
        raise ValueError("corpus body ids must be unique")
    by_class = {name: [body for body in bodies if body["class"] == name] for name in EXPECTED_HISTOGRAMS}
    for name, expected in EXPECTED_HISTOGRAMS.items():
        rows = by_class[name]
        histogram = dict(sorted(Counter(int(row["segment_count"]) for row in rows).items()))
        if histogram != expected:
            raise ValueError(f"{name} histogram {histogram} != {expected}")
        for row in rows:
            if len(row["segments"]) != row["segment_count"]:
                raise ValueError(f"{row['id']} segment count mismatch")
            for expected_slot, seg in enumerate(row["segments"], start=1):
                if seg["slot"] != expected_slot or not 0 <= seg["parent"] < expected_slot:
                    raise ValueError(f"{row['id']} has invalid slot/parent ordering")
                if seg["depth"] > 5:
                    raise ValueError(f"{row['id']} exceeds max depth")
    h1 = by_class["H1"]
    h2 = by_class["H2"]
    if sum(row["near_zero_thrust_control"] for row in h1) < 8:
        raise ValueError("H1 must contain at least eight near-zero-thrust controls")
    if sum(row["near_zero_thrust_control"] for row in h2) < 8:
        raise ValueError("H2 must contain at least eight near-zero-thrust controls")
    if sum(row["has_mirrored_branch"] for row in h1) != 32:
        raise ValueError("H1 must have exactly 32 mirrored bodies")
    if sum(row["fin_active"] for row in h1) != 26:
        raise ValueError("H1 must have exactly 26 fin tails")
    if sum(row["tilted_anisotropic"] for row in h1) != 32:
        raise ValueError("H1 must have exactly 32 tilted bodies")
    if sum(row["fin_active"] for row in h2) != 32:
        raise ValueError("H2 must have exactly 32 fin tails")
    full_h2 = [row for row in h2 if row["segment_count"] == 16]
    if len(full_h2) != 8 or not all(row["tilted_anisotropic"] for row in full_h2):
        raise ValueError("all eight full H2 bodies must be tilted anisotropic")
    full_indices = [index for index, row in enumerate(h2) if row["segment_count"] == 16]
    if full_indices[-1] - full_indices[0] < 48 or max(
        right - left for left, right in pairwise(full_indices)
    ) > 9:
        raise ValueError("H2 full bodies must be interleaved across the class")
    if any(h2[index]["fin_active"] == h2[index + 1]["fin_active"] for index in range(63)):
        raise ValueError("H2 fin and non-fin rows must be interleaved")
    scale_signatures = {
        tuple(round(value, 6) for value in row["segments"][0]["abc_m"]) for row in full_h2
    }
    aspect_signatures = {
        round(row["segments"][0]["abc_m"][0] / row["segments"][0]["abc_m"][2], 6)
        for row in full_h2
    }
    if len(scale_signatures) != 8 or len(aspect_signatures) != 8:
        raise ValueError("H2 full bodies must occupy eight distinct scale/aspect cases")


def verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    if digest != sidecar:
        raise ValueError("corpus sha256 sidecar mismatch")
    return digest
