import torch

from sirrobin.physics.energy import prefix_budget


def test_prefix_budget_catches_endpoint_cancellation_and_bias():
    residuals = torch.tensor([1e-6] * 100 + [-1e-6] * 100, dtype=torch.float64)
    scales = torch.ones_like(residuals)
    budget = prefix_budget(residuals, scales, warmup=0)
    assert budget.max_normalized_after_warmup > abs(float(residuals.sum() / scales.sum()))
    biased = prefix_budget(torch.full((200,), 1e-6, dtype=torch.float64), scales, warmup=10)
    assert biased.monotone_bias
