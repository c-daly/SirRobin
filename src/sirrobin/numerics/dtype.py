"""Deliberate f32-hot/f64-reference dtype policy."""

import torch

HOT_DTYPE = torch.float32
REFERENCE_DTYPE = torch.float64
INDEX_DTYPE = torch.int64


def require_float_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"expected float32 or float64, got {dtype}")
