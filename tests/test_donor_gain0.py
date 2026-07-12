import json
from pathlib import Path

import pytest
import torch

from sirrobin.benchmarks.episode import run_episode
from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch
from sirrobin.physics.mass_matrix import prepare_mass_data
from sirrobin.physics.pose import resolve_pose
from sirrobin.physics.swim_step import SwimKernel


def _fixtures():
    corpus = json.loads(Path("oracle/fixtures/corpus.json").read_text())
    donor = json.loads(Path("oracle/fixtures/gain0_donor.json").read_text())
    return corpus, donor


def test_untouched_donor_fixture_provenance_and_reconstruction():
    corpus, donor = _fixtures()
    assert donor["schema"] == "sirrobin.locomotion.gain0.v1"
    assert len(donor["donor_sha256"]) == 64
    assert donor["corpus_sha256"] == Path("oracle/fixtures/corpus.sha256").read_text().split()[0]
    rows = corpus["bodies"]
    config = LocomotionConfig(ellipsoid_mass_gain=0.0, fin_plane_gain=0.0)
    body = BodyBatch.from_rows(rows, config, dtype=torch.float64)
    pose = resolve_pose(body, body.gait_time, apply_gait=False)
    static = prepare_mass_data(body, config)
    by_id = {row["id"]: row for row in donor["bodies"]}
    for bi, row in enumerate(rows):
        expected = by_id[row["id"]]
        assert expected["tail"] + 1 == row["segment_count"]
        for seg in expected["segments"]:
            slot = seg["slot"]
            assert torch.allclose(
                pose.pos[bi, slot], torch.tensor(seg["rest_pos"], dtype=torch.float64), rtol=0, atol=2e-6
            )
            actual_q = pose.rot[bi, slot]
            expected_q = torch.tensor(seg["rest_rot"], dtype=torch.float64)
            assert abs(float(torch.dot(actual_q, expected_q))) > 1 - 2e-6
            assert abs(float(static.seg_mass_sim[bi, slot]) - seg["mass_sim"]) < 2e-6
            assert torch.allclose(
                static.added_mass[bi, slot],
                torch.tensor(seg["added_mass_kg"], dtype=torch.float64),
                rtol=2e-4,
                atol=2e-6,
            )


def test_gain0_h0_episode_matches_untouched_donor_aggregate():
    corpus, donor = _fixtures()
    row = next(row for row in corpus["bodies"] if row["id"] == "H0-00")
    expected = next(row for row in donor["bodies"] if row["id"] == "H0-00")["aggregate"]
    config = LocomotionConfig(n_cap=1, n_live=1, ellipsoid_mass_gain=0.0, fin_plane_gain=0.0)
    body = BodyBatch.from_rows([row], config, dtype=torch.float64)
    actual = run_episode(SwimKernel(body, config))
    dtype = actual.cruise_speed.dtype
    assert torch.allclose(
        actual.cruise_speed,
        torch.tensor([expected["cruiseSpeed"]], dtype=dtype),
        rtol=1e-3,
        atol=1e-5,
    )
    assert torch.allclose(
        actual.cost_of_transport,
        torch.tensor([expected["costOfTransport"]], dtype=dtype),
        rtol=1e-3,
        atol=1e-4,
    )
    assert torch.allclose(
        actual.reactive_ratio,
        torch.tensor([expected["reactiveRatio"]], dtype=dtype),
        rtol=1e-3,
        atol=1e-4,
    )


@pytest.mark.parametrize("body_id", ["H1-33", "H2-25"])
def test_gain0_bug_inert_h1_h2_episodes_match_untouched_donor(body_id):
    corpus, donor = _fixtures()
    row = next(row for row in corpus["bodies"] if row["id"] == body_id)
    assert not row["tilted_anisotropic"]
    expected = next(row for row in donor["bodies"] if row["id"] == body_id)["aggregate"]
    config = LocomotionConfig(
        n_cap=1,
        n_live=1,
        ellipsoid_mass_gain=0.0,
        fin_plane_gain=0.0,
    )
    actual = run_episode(SwimKernel(BodyBatch.from_rows([row], config, dtype=torch.float64), config))
    expected_values = torch.tensor(
        [expected["cruiseSpeed"], expected["costOfTransport"], expected["reactiveRatio"]],
        dtype=torch.float64,
    )
    actual_values = torch.stack(
        (actual.cruise_speed[0], actual.cost_of_transport[0], actual.reactive_ratio[0])
    )
    assert torch.allclose(actual_values, expected_values, rtol=1e-3, atol=1e-4)


def test_gain0_h0_first_32_steps_match_untouched_donor_trace():
    corpus, donor = _fixtures()
    row = next(row for row in corpus["bodies"] if row["id"] == "H0-00")
    expected = next(row for row in donor["bodies"] if row["id"] == "H0-00")["trace"]
    for dtype in (torch.float64, torch.float32):
        config = LocomotionConfig(
            n_cap=1,
            n_live=1,
            ellipsoid_mass_gain=0.0,
            fin_plane_gain=0.0,
        )
        body = BodyBatch.from_rows([row], config, dtype=dtype)
        kernel = SwimKernel(body, config)
        for trace_row in expected:
            kernel.step()
            assert torch.allclose(
                body.v_com[0],
                torch.tensor(trace_row["v_com"], dtype=dtype),
                rtol=1e-4,
                atol=2e-6,
            )
            assert torch.allclose(
                body.x_com[0],
                torch.tensor(trace_row["x_origin"], dtype=dtype),
                rtol=1e-4,
                atol=1e-7,
            )
