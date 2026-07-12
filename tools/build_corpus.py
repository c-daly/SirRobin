#!/usr/bin/env python3
"""Build the one frozen S0 developed-body corpus.

This script is an offline authoring tool. Tests consume the committed JSON and
must never invoke this generator to obtain expected values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "oracle" / "fixtures" / "corpus.json"
HASH_OUT = OUT.with_suffix(".sha256")

H1_COUNTS = (
    [2] * 6 + [3] * 8 + [4] * 8 + [5] * 8 + [6] * 8 + [7] * 8 + [8] * 6 + [10] * 4 + [12] * 4 + [16] * 4
)
H2_FULL_INDICES = (3, 10, 19, 26, 35, 42, 51, 58)
_h2_short = iter([2, 3] * 28)
H2_COUNTS = [16 if index in H2_FULL_INDICES else next(_h2_short) for index in range(64)]


def segment(
    body_index: int,
    slot: int,
    count: int,
    *,
    mirrored: bool,
    tilted: bool,
    fin: bool,
    control: bool,
    swim_wave: float,
) -> dict:
    root = slot == 1
    if root:
        parent = 0
        depth = 0
    elif mirrored and slot in (2, 3):
        parent = 1
        depth = 1
    elif mirrored and slot <= 6:
        parent = slot - 1
        depth = slot - 2
    elif not mirrored and slot <= 6:
        parent = slot - 1
        depth = slot - 1
    else:
        parent = 1
        depth = 1
    side = -1.0 if slot % 2 == 0 else 1.0
    shape_slot = slot if not mirrored or slot % 2 == 0 else slot - 1
    quartile = body_index % 4
    if count == 16 and body_index in H2_FULL_INDICES:
        octile = H2_FULL_INDICES.index(body_index)
        scale = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40)[octile]
        aspect_a = 0.82 + 0.05 * octile
        aspect_c = 1.18 - 0.04 * octile
    else:
        scale = (0.75, 0.95, 1.15, 1.35)[quartile]
        aspect_a = aspect_c = 1.0
    tail = slot == count
    surface = tail and fin
    a = max(
        0.06,
        round((0.10 + 0.012 * ((body_index + shape_slot) % 5)) * scale * aspect_a, 6),
    )
    b = max(0.06, round((0.075 + 0.010 * ((2 * body_index + shape_slot) % 4)) * scale, 6))
    c = max(
        0.15,
        round((0.18 + 0.025 * ((body_index + 3 * shape_slot) % 5)) * scale * aspect_c, 6),
    )
    if surface:
        b = 0.05
        a = round((0.18 + 0.05 * quartile) * scale, 6)
        c = max(0.15, round((0.12 + 0.03 * ((body_index // 4) % 4)) * scale, 6))
    x = 0.0 if root else round((0.06 + 0.015 * (slot % 3)) * side if mirrored else 0.012 * side, 6)
    y = 0.0 if root else round(0.01 * (((body_index + shape_slot) % 3) - 1), 6)
    z = 0.0 if root else (2.0 if tail else round(0.22 + 0.018 * (shape_slot % 4), 6))
    pitch = 0.0 if not tilted or root else float((-18, -7, 9, 21)[(body_index + shape_slot) % 4])
    roll_base = 0.0 if not tilted or root else float((-16, -5, 8, 19)[(2 * body_index + shape_slot) % 4])
    yaw_base = 0.0 if root else float((-12, -4, 5, 14)[(body_index + 2 * shape_slot) % 4])
    roll = roll_base * (side if mirrored else 1.0)
    yaw = yaw_base * (side if mirrored else 1.0)
    amp = 0.0 if root or control else float((8, 20, 36, 52)[(body_index + slot) % 4])
    return {
        "slot": slot,
        "parent": parent,
        "depth": depth,
        "local_pos_m": [x, y, z],
        "local_euler_deg_xyz": [pitch, yaw, roll],
        "abc_m": [a, b, c],
        "density_gene_sim_mass_m3": float((2.0, 3.5, 5.0, 6.5)[(body_index + slot) % 4]),
        "amp_deg": amp,
        "phase_rad": round(-depth * swim_wave, 8),
        "is_surface": surface,
        "is_tail": tail,
        "fin_span_m": round(2.0 * a, 6) if surface else 0.0,
        "fin_chord_m": round(2.0 * c, 6) if surface else 0.0,
    }


def body(class_name: str, index: int, count: int) -> dict:
    if class_name == "H0":
        mirrored = tilted = fin = control = False
        freq, wave = 1.0, 1.0
    elif class_name == "H1":
        mirrored = index < 32
        tilted = index < 32
        fin = index < 26
        control = index % 8 == 0
        freq = (0.55, 0.95, 1.4, 1.9)[index % 4]
        wave = (0.45, 0.85, 1.25, 1.75)[(index // 4) % 4]
    else:
        mirrored = index % 2 == 0
        tilted = count == 16 or index < 24
        fin = index % 2 == 0
        control = index % 8 == 1
        freq = (0.5, 1.0, 1.5, 2.0)[index % 4]
        wave = (0.4, 0.9, 1.4, 1.9)[(index // 4) % 4]
    return {
        "id": f"{class_name}-{index:02d}",
        "class": class_name,
        "segment_count": count,
        "has_mirrored_branch": mirrored,
        "tilted_anisotropic": tilted,
        "fin_active": fin,
        "near_zero_thrust_control": control,
        "swim_freq_hz": freq,
        "swim_wave_rad_per_depth": wave,
        "segments": [
            segment(
                index,
                slot,
                count,
                mirrored=mirrored,
                tilted=tilted,
                fin=fin,
                control=control,
                swim_wave=wave,
            )
            for slot in range(1, count + 1)
        ],
    }


def main() -> None:
    h0_template = body("H0", 0, 6)
    bodies = []
    for i in range(64):
        row = json.loads(json.dumps(h0_template))
        row["id"] = f"H0-{i:02d}"
        bodies.append(row)
    bodies.extend(body("H1", i, count) for i, count in enumerate(H1_COUNTS))
    bodies.extend(body("H2", i, count) for i, count in enumerate(H2_COUNTS))
    payload = {
        "schema": "sirrobin.locomotion.corpus.v1",
        "authoring_note": "Literal developed bodies; do not regenerate during tests.",
        "classes": {"H0": 64, "H1": 64, "H2": 64},
        "bodies": bodies,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    OUT.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    HASH_OUT.write_text(f"{digest}  {OUT.name}\n", encoding="ascii")
    print(f"wrote {OUT} ({len(bodies)} bodies), sha256={digest}")


if __name__ == "__main__":
    main()
