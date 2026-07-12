import torch

from sirrobin.numerics.quat import euler_unity_deg, multiply, rotate


def test_identity_and_axis_rotation():
    q = euler_unity_deg(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float64))
    v = rotate(q, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(v, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)
    identity = euler_unity_deg(torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(multiply(identity, q), q, atol=1e-12)
