#!/usr/bin/env python3
"""Independent numpy/scipy gain1 oracle.

Forbidden imports: torch, sirrobin production packages, donor C# code. The
committed output is generated once and consumed as literal expected data.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "oracle" / "fixtures" / "corpus.json"
OUT = ROOT / "oracle" / "fixtures" / "gain1_analytic.json"
QUAD_OUT = ROOT / "oracle" / "fixtures" / "quadrature_gl256.json"
QUAD32_OUT = ROOT / "oracle" / "fixtures" / "quadrature_gl32_negative.json"

RHO_WATER = 1000.0
KG_PER_SIM_MASS = 250.0
DT = 1.0 / 120.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gl(order: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(order)
    return (x + 1.0) * 0.5, w * 0.5


def lamb_coeff(abc: np.ndarray, order: int) -> np.ndarray:
    t, weights = gl(order)
    axes2 = abc**2
    scale = float(np.max(axes2))
    lam = scale * (1.0 - t) / t
    jac = scale / t**2
    shifted = axes2[:, None] + lam[None, :]
    delta = np.sqrt(np.prod(shifted, axis=0))
    integral = np.sum(weights[None, :] * jac[None, :] / (shifted * delta[None, :]), axis=1)
    coeff = np.prod(abc) * integral
    return coeff * (2.0 / np.sum(coeff))


def lamb_quad(abc: np.ndarray) -> np.ndarray:
    axes2 = abc**2
    product = float(np.prod(abc))
    out = []
    for axis2 in axes2:

        def f(lam: float, axis2_: float = float(axis2)) -> float:
            delta = math.sqrt(float(np.prod(axes2 + lam)))
            return product / ((axis2_ + lam) * delta)

        out.append(quad(f, 0.0, np.inf, epsabs=1e-13, epsrel=1e-13, limit=500)[0])
    coeff = np.asarray(out)
    return coeff * (2.0 / np.sum(coeff))


def factors(coeff: np.ndarray) -> np.ndarray:
    return coeff / (2.0 - coeff)


def added_mass(abc: np.ndarray) -> np.ndarray:
    volume = 4.0 / 3.0 * math.pi * float(np.prod(abc))
    return factors(lamb_coeff(abc, 256)) * RHO_WATER * volume


def unity_quat(euler: list[float]) -> Rotation:
    # Unity applies z, then x, then y; scipy uppercase denotes intrinsic axes.
    x, y, z = euler
    rz = Rotation.from_rotvec(np.array([0.0, 0.0, math.radians(z)]))
    rx = Rotation.from_rotvec(np.array([math.radians(x), 0.0, 0.0]))
    ry = Rotation.from_rotvec(np.array([0.0, math.radians(y), 0.0]))
    return ry * rx * rz


def body_static(row: dict) -> dict:
    rotations: list[Rotation] = [Rotation.identity()]
    positions = [np.zeros(3)]
    total_mass_sim = 0.0
    matrix_added = np.zeros((3, 3))
    segment_outputs = []
    for seg in row["segments"]:
        parent = int(seg["parent"])
        local = unity_quat(seg["local_euler_deg_xyz"])
        theta = float(seg["amp_deg"]) * math.sin(float(seg["phase_rad"]))
        flex = Rotation.from_rotvec(np.array([0.0, math.radians(theta), 0.0]))
        rot = rotations[parent] * local * flex
        pos = positions[parent] + rotations[parent].apply(np.asarray(seg["local_pos_m"], dtype=float))
        rotations.append(rot)
        positions.append(pos)
        abc = np.asarray(seg["abc_m"], dtype=float)
        mass_sim = (
            max(0.1, 8.0 * float(np.prod(abc)) * float(seg["density_gene_sim_mass_m3"])) * math.pi / 6.0
        )
        madd = added_mass(abc)
        if seg["is_surface"] and seg["fin_span_m"] > 0:
            true_abc = np.array([abc[0], 0.5 * seg["fin_span_m"], abc[2]])
            madd[1] = added_mass(true_abc)[0]
        rotation_matrix = rot.as_matrix()
        matrix_added += rotation_matrix @ np.diag(madd) @ rotation_matrix.T
        total_mass_sim += mass_sim
        segment_outputs.append(
            {
                "slot": seg["slot"],
                "position_m": pos.tolist(),
                "rotation_xyzw": rot.as_quat().tolist(),
                "mass_sim": mass_sim,
                "added_mass_kg": madd.tolist(),
            }
        )
    matrix = matrix_added + np.eye(3) * total_mass_sim * KG_PER_SIM_MASS
    impulse = np.array([0.2, 0.0, -0.1]) * DT
    xz = matrix[np.ix_([0, 2], [0, 2])]
    dv_xz = np.linalg.solve(xz, impulse[[0, 2]])
    return {
        "id": row["id"],
        "mass_sim": total_mass_sim,
        "mass_kg": total_mass_sim * KG_PER_SIM_MASS,
        "matrix_kg": matrix.tolist(),
        "test_impulse_ns": impulse.tolist(),
        "constrained_dv_m_s": [float(dv_xz[0]), 0.0, float(dv_xz[1])],
        "segments": segment_outputs,
    }


def force_cases() -> list[dict]:
    rows = []
    for i, (mt, u, vt, slope) in enumerate(
        ((2.0, 0.3, 0.1, 0.2), (0.7, 0.0, 0.4, -0.1), (1.2, 0.8, -0.25, 0.15))
    ):
        wt = vt + u * slope
        t_react = 0.5 * mt * (vt**2 - u**2 * slope**2)
        p_wake = 0.5 * mt * u * wt**2
        p_input = mt * u * vt * wt
        rows.append(
            {
                "id": f"reactive-{i}",
                "inputs": [mt, u, vt, slope],
                "thrust_n": t_react,
                "wake_w": p_wake,
                "input_w": p_input,
            }
        )
    return rows


def fin_cases() -> list[dict]:
    rows = []
    inputs = (
        (4.0, 0.12, 0.8, 0.18, 0.12),
        (2.5, 0.08, 0.0, 0.30, -0.08),
        (6.0, 0.20, -0.4, 0.25, 0.20),
        (3.0, 0.10, 0.0, 0.0, 0.0),
    )
    for index, (aspect_ratio, area, u, vt, slope) in enumerate(inputs):
        lift_slope = 2.0 * math.pi * aspect_ratio / (aspect_ratio + 2.0)
        u_cl = max(0.0, u)
        speed2 = u_cl**2 + vt**2
        if speed2 < 1e-8:
            thrust = input_power = wake_power = 0.0
        else:
            speed = math.sqrt(speed2)
            alpha = max(-0.35, min(0.35, math.atan2(vt, u_cl) - math.asin(max(-1.0, min(1.0, slope)))))
            cl = lift_slope * alpha
            dynamic = 0.5 * RHO_WATER * u_cl**2 * area
            lift = dynamic * cl
            cdi = 0.02 + cl**2 / (math.pi * 0.9 * max(aspect_ratio, 1e-4))
            drag = dynamic * cdi
            sin_beta = vt / speed
            cos_beta = u_cl / speed
            thrust = lift * sin_beta - drag * cos_beta
            normal_force = lift * cos_beta + drag * sin_beta
            input_power = normal_force * vt
            wake_power = drag * speed
        rows.append(
            {
                "id": f"fin-{index}",
                "inputs": [lift_slope, aspect_ratio, area, u, vt, slope],
                "thrust_n": thrust,
                "input_w": input_power,
                "wake_w": wake_power,
            }
        )
    return rows


def main() -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    t256, w256 = gl(256)
    t32, w32 = gl(32)
    QUAD_OUT.write_text(
        json.dumps({"order": 256, "nodes_0_1": t256.tolist(), "weights_0_1": w256.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )
    QUAD32_OUT.write_text(
        json.dumps(
            {
                "order": 32,
                "nodes_0_1": t32.tolist(),
                "weights_0_1": w32.tolist(),
                "status": "negative-regression-only; fails 1e-8 convergence on 10:1 corpus",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lamb_cases = []
    shapes = {
        "sphere": [0.5, 0.5, 0.5],
        "prolate_10_1": [0.05, 0.05, 0.5],
        "oblate_10_1": [0.5, 0.5, 0.05],
        "triaxial": [0.42, 0.17, 0.08],
    }
    for name, raw in shapes.items():
        abc = np.asarray(raw, dtype=float)
        c32 = lamb_coeff(abc, 32)
        c256 = lamb_coeff(abc, 256)
        c512 = lamb_coeff(abc, 512)
        cq = lamb_quad(abc)
        lamb_cases.append(
            {
                "id": name,
                "abc_m": raw,
                "coeff": c256.tolist(),
                "factor": factors(c256).tolist(),
                "gl256_vs_gl512_max_rel": float(
                    np.max(np.abs(c256 - c512) / np.maximum(np.abs(c512), 1e-30))
                ),
                "gl256_vs_quad_max_rel": float(np.max(np.abs(c256 - cq) / np.maximum(np.abs(cq), 1e-30))),
                "gl32_negative_max_rel": float(np.max(np.abs(c32 - cq) / np.maximum(np.abs(cq), 1e-30))),
            }
        )
    selected = [
        row
        for row in corpus["bodies"]
        if row["id"]
        in {"H0-00", "H1-00", "H1-31", "H1-63", "H2-00", "H2-56", "H2-58", "H2-63"}
    ]
    payload = {
        "schema": "sirrobin.locomotion.gain1.v1",
        "corpus_sha256": sha(CORPUS),
        "generator_sha256": sha(Path(__file__)),
        "constants": {"rho_water_kg_m3": RHO_WATER, "kg_per_sim_mass": KG_PER_SIM_MASS, "dt_s": DT},
        "lamb_cases": lamb_cases,
        "reactive_cases": force_cases(),
        "fin_cases": fin_cases(),
        "body_cases": [body_static(row) for row in selected],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}, sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
