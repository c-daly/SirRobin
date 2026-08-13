import json
from pathlib import Path

import pytest
import torch

from sirrobin.core.live_world import advance_live_world, initialize_live_state
from sirrobin.fields.geometry import GridGeometry
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.live_step import step_live

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def _rows():
    bodies = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    ids = {"swimmer", "mirrored", "wide-16"}
    return [body for body in bodies if body["id"] in ids]


def test_cuda_development_matches_cpu_and_live_step_has_no_interventions():
    genotype_cpu = GenotypeBatch.from_donor_rows(_rows(), dtype=torch.float32)
    body_cpu = develop(genotype_cpu)
    body_cuda = develop(genotype_cpu.to("cuda"))
    assert torch.equal(body_cpu.seg_mask, body_cuda.seg_mask.cpu())
    assert torch.equal(body_cpu.tail_slot, body_cuda.tail_slot.cpu())
    assert torch.allclose(
        body_cpu.added_mass_flu_kg,
        body_cuda.added_mass_flu_kg.cpu(),
        rtol=2e-5,
        atol=2e-6,
    )
    state = initialize_live_state(body_cuda)
    fluid = FluidSample(
        torch.full(body_cuda.alive.shape, 1000.0, device="cuda"),
        torch.zeros((*body_cuda.alive.shape, 3), device="cuda"),
    )
    for _ in range(20):
        ledger = step_live(body_cuda, state, fluid, LiveLocomotionConfig())
    assert not torch.any(ledger.solve_regularized).item()
    assert not torch.any(ledger.yaw_inertia_floor_hit).item()
    assert not torch.any(ledger.omega_backstop_hit).item()
    assert not torch.any(ledger.nonfinite).item()
    assert torch.isfinite(state.velocity_rel_water_enu_m_s).all().item()


def test_cuda_graph_homing_settles_without_yaw_backstop():
    genotype = GenotypeBatch.from_donor_rows([_rows()[0]], dtype=torch.float32, device="cuda")
    body = develop(genotype)
    state = initialize_live_state(body)
    fluid = FluidSample(
        torch.full(body.alive.shape, 1000.0, device="cuda"),
        torch.zeros((*body.alive.shape, 3), device="cuda"),
    )
    config = LiveLocomotionConfig()
    geometry = GridGeometry(8, 8, 4, 100.0, 100.0, 40.0)
    requested = torch.tensor([[[0.0, 1.0]]], device="cuda")

    def step():
        return advance_live_world(
            body,
            state,
            fluid,
            config,
            geometry,
            requested_heading_enu=requested,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ledger = step()
    for _ in range(2396):
        graph.replay()
    torch.cuda.synchronize()
    error = torch.atan2(
        torch.sin(0.5 * torch.pi - state.yaw_rad),
        torch.cos(0.5 * torch.pi - state.yaw_rad),
    ).abs()
    assert error.item() < torch.deg2rad(torch.tensor(15.0)).item()
    assert not ledger.omega_backstop_hit.any().item()
