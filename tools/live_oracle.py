"""Independent NumPy-only S2 frame/development/mechanics oracle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DONOR_FIXTURE = ROOT / "oracle/fixtures/live/donor_development_live.json"
OUTPUT = ROOT / "oracle/fixtures/live/gain1_canonical.json"

# v_flu = DONOR_TO_FLU @ v_donor. Donor posterior +z becomes FLU aft -x.
DONOR_TO_FLU = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(512)
GL_T = 0.5 * (GL_NODES + 1.0)
GL_W = 0.5 * GL_WEIGHTS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unity_euler_matrix(deg: list[float]) -> np.ndarray:
    x, y, z = np.radians(np.asarray(deg, dtype=np.float64))
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return ry @ rx @ rz


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = matrix
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                          (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                          (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                          0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def lamb_added_mass(axes: np.ndarray, rho: float = 1000.0) -> np.ndarray:
    t, w = GL_T, GL_W
    scale = float(np.max(axes * axes))
    lam = scale * t / (1.0 - t)
    jac = scale / (1.0 - t) ** 2
    delta = np.sqrt(np.prod(axes[:, None] ** 2 + lam[None, :], axis=0))
    alpha = np.array([
        np.prod(axes) * np.sum(w * jac / ((axis * axis + lam) * delta)) for axis in axes
    ])
    alpha *= 2.0 / alpha.sum()
    k = alpha / (2.0 - alpha)
    volume = 4.0 * math.pi * np.prod(axes) / 3.0
    return k * rho * volume


def canonical_segment(gene: dict, side: float, parent_axes: np.ndarray) -> dict:
    size = np.asarray(gene["size"], dtype=np.float64)
    surface = gene["type"] == "Surface"
    # P1 reinterpretation: preserve all three donor dimensions as one ellipsoid. A Surface bit activates
    # lift but does not force a second hidden thickness/span geometry.
    sc_donor = np.array([max(0.12, size[0]), max(0.12, size[1]), max(0.3, size[2])])
    axes = np.array([sc_donor[2], sc_donor[0], sc_donor[1]]) * 0.5
    local_donor = np.array([gene["attach"][0] * side, gene["attach"][1], gene["attach"][2]])
    local_pos = DONOR_TO_FLU @ local_donor
    euler = [gene["orient_deg"][0], gene["orient_deg"][1] * side,
             gene["orient_deg"][2] * side]
    rot_donor = unity_euler_matrix(euler)
    local_rot = DONOR_TO_FLU @ rot_donor @ DONOR_TO_FLU.T
    volume = 4.0 * math.pi * float(np.prod(axes)) / 3.0
    mass = volume * float(gene["density"])
    drag_area = np.array([4 * axes[1] * axes[2], 4 * axes[0] * axes[2], 4 * axes[0] * axes[1]])
    added = lamb_added_mass(axes)
    fin_perp = float(added[1]) if surface else 0.0
    return {
        "local_pos": local_pos,
        "local_rot_matrix": local_rot,
        "local_rot": matrix_to_quat_xyzw(local_rot),
        "axes": axes,
        "volume": volume,
        "mass_sim": mass,
        "drag_area": drag_area,
        "added_mass_kg": added,
        "fin_perpendicular_kg": fin_perp,
        "surface": surface,
        "intake": gene["port"] == "Intake",
        "joint_amp_rad": math.radians(min(58.0, max(0.0, float(gene["joint_amp_deg"])))),
    }


def develop_tree(body: dict) -> dict:
    genome = body["genotype"]
    segments: list[dict] = []

    def walk(gene: dict, parent: int, depth: int, side: float, parent_pos: np.ndarray,
             parent_rot: np.ndarray, parent_axes: np.ndarray) -> None:
        if len(segments) >= 16 or depth > 5:
            return
        seg = canonical_segment(gene, side, parent_axes)
        rest_pos = parent_pos + parent_rot @ seg["local_pos"]
        rest_rot = parent_rot @ seg["local_rot_matrix"]
        index = len(segments)
        segments.append({**seg, "parent": parent, "depth": depth, "side": side,
                         "rest_pos": rest_pos, "rest_rot": matrix_to_quat_xyzw(rest_rot),
                         "rest_rot_matrix": rest_rot,
                         "phase_rad": -depth * float(genome["swim_wave_rad_per_depth"])})
        for child in gene["children"]:
            walk(child, index, depth + 1, side, rest_pos, rest_rot, seg["axes"])
            if child["mirror"]:
                walk(child, index, depth + 1, -side, rest_pos, rest_rot, seg["axes"])

    walk(genome["root"], -1, 0, 1.0, np.zeros(3), np.eye(3), np.ones(3))
    endpoints = [s["rest_pos"] + s["rest_rot_matrix"] @ np.array([-s["axes"][0], 0, 0])
                 for s in segments]
    tail = max(range(len(endpoints)), key=lambda i: (-endpoints[i][0], i))
    mass_total = sum(s["mass_sim"] for s in segments)
    com = sum((s["mass_sim"] * s["rest_pos"] for s in segments), np.zeros(3)) / mass_total

    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    cleaned = []
    for s in segments:
        cleaned.append({k: serial(v) for k, v in s.items() if not k.endswith("_matrix")})
    return {"id": body["id"], "tail": tail, "tail_tip_rest": endpoints[tail].tolist(),
            "com_rest": com.tolist(), "tail_aft_dot": float((endpoints[tail] - com)[0]),
            "segments": cleaned}


def yaw_cases() -> dict:
    inertia, coeff, omega0 = 12.0, 3.0, 1.5
    times = np.linspace(0.0, 4.0, 17)
    omega = omega0 / (1.0 + coeff * abs(omega0) * times / inertia)
    work = []
    for l0, l1, i0, i1 in [(3.0, 3.4, 2.0, 2.2), (-4.0, -3.1, 5.0, 4.4), (0.2, -0.1, 0.8, 1.1)]:
        w0, w1 = l0 / i0, l1 / i1
        dk = l1 * l1 / (2 * i1) - l0 * l0 / (2 * i0)
        impulse = 0.5 * (w0 + w1) * (l1 - l0)
        delta_i = 0.5 * l0 * l1 * (1 / i1 - 1 / i0)
        work.append({"inputs": [l0, l1, i0, i1], "delta_ke": dk,
                     "work_impulse": impulse, "work_delta_inertia": delta_i,
                     "residual": dk - impulse - delta_i})
    return {"quadratic_decay": {"inertia": inertia, "coefficient": coeff, "omega0": omega0,
                                 "times": times.tolist(), "omega": omega.tolist()},
            "angular_work": work}


def main() -> None:
    donor = json.loads(DONOR_FIXTURE.read_text(encoding="utf-8"))
    result = {
        "schema": "sirrobin.development-live.gain1.v1",
        "generator_sha256": sha(Path(__file__)),
        "donor_fixture_sha256": sha(DONOR_FIXTURE),
        "frame": {"donor_to_flu": DONOR_TO_FLU.tolist(), "determinant": float(np.linalg.det(DONOR_TO_FLU))},
        "bodies": [develop_tree(body) for body in donor["bodies"]],
        "yaw_cases": yaw_cases(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(result['bodies'])} bodies)")


if __name__ == "__main__":
    main()
