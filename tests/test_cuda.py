import json
from pathlib import Path

import pytest
import torch

from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch
from sirrobin.physics.swim_step import SwimKernel

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]


def _rows():
    corpus = json.loads(Path("oracle/fixtures/corpus.json").read_text())
    by_id = {row["id"]: row for row in corpus["bodies"]}
    return [by_id[body_id] for body_id in ("H1-00", "H1-31", "H2-03", "H2-58")]


def test_cuda_single_step_matches_cpu_f32_and_never_regularizes():
    rows = _rows()
    config = LocomotionConfig(n_cap=len(rows), n_live=len(rows))
    cpu_body = BodyBatch.from_rows(rows, config, dtype=torch.float32)
    gpu_body = BodyBatch.from_rows(rows, config, dtype=torch.float32, device="cuda")
    cpu_ledger = SwimKernel(cpu_body, config).step()
    gpu_ledger = SwimKernel(gpu_body, config).step()
    assert torch.allclose(gpu_body.v_com.cpu(), cpu_body.v_com, rtol=1e-4, atol=2e-6)
    assert torch.allclose(gpu_ledger.m_after.cpu(), cpu_ledger.m_after, rtol=1e-4, atol=2e-4)
    assert not gpu_ledger.regularized.any()


def test_cuda_graph_replay_advances_in_place_state():
    rows = _rows()
    config = LocomotionConfig(n_cap=len(rows), n_live=len(rows))
    body = BodyBatch.from_rows(rows, config, dtype=torch.float32, device="cuda")
    kernel = SwimKernel(body, config)
    addresses = (body.v_com.data_ptr(), body.x_com.data_ptr(), body.gait_time.data_ptr())
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            kernel.step()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ledger = kernel.step()
    time_before = body.gait_time.clone()
    position_before = body.x_com.clone()
    for _ in range(32):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.all(body.gait_time > time_before)
    assert torch.any(body.x_com != position_before)
    assert (body.v_com.data_ptr(), body.x_com.data_ptr(), body.gait_time.data_ptr()) == addresses
    assert not ledger.regularized.any()
