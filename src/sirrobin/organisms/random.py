"""Counter-based deterministic words for device organism transactions."""

from __future__ import annotations

import torch

_PRIME = 2_147_483_647


def identity_word_u31(
    stable_id: torch.Tensor,
    world_index: torch.Tensor,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Return a reproducible positive 31-bit word for each stable identity.

    All intermediate products remain below signed-int64 overflow. The function is
    counter based: population order, slot reuse, and the number of other random
    requests cannot change an identity's word.
    """

    seed_term = seed % _PRIME
    stream_term = stream % _PRIME
    value = torch.remainder(stable_id, _PRIME)
    value = torch.remainder(
        value
        + seed_term
        + (world_index + 1) * 104_729
        + (stream_term + 1) * 130_363,
        _PRIME,
    )
    value = torch.where(value == 0, 1, value)
    value = torch.remainder(value * 1_103_515_245 + 12_345, _PRIME)
    value = torch.remainder(value * 1_664_525 + 1_013_904_223, _PRIME)
    value = torch.remainder(value * 22_695_477 + 1, _PRIME)
    return torch.where(value == 0, 1, value)


def identity_uniform(
    stable_id: torch.Tensor,
    world_index: torch.Tensor,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Map an identity word to float64 in the half-open interval [0,1)."""

    word = identity_word_u31(
        stable_id,
        world_index,
        seed=seed,
        stream=stream,
    )
    return (word.to(torch.float64) - 1.0) / (_PRIME - 1)
