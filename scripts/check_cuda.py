#!/usr/bin/env python3
"""Fail-fast CUDA compatibility check for the production uv environment."""

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to this Python process")
x = torch.arange(1024, device="cuda", dtype=torch.float32)
y = (x.square().sum()).item()
print(
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "result": y,
    }
)
