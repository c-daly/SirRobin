"""Fixed-schedule static-buffer lifecycle workload."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.physics.contracts import BodyBatch


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    step: int
    slots: tuple[int, ...]
    old_ids: tuple[int, ...]
    new_ids: tuple[int, ...]


def apply_fixed_churn(
    body: BodyBatch, step: int, *, period: int = 1000, fraction: float = 0.02
) -> LifecycleEvent | None:
    if step <= 0 or step % period != 0:
        return None
    live_slots = torch.nonzero(body.alive, as_tuple=False).flatten()
    count = max(1, int(live_slots.numel() * fraction))
    # Rotate which lanes churn without stochastic state or host-dependent ordering.
    offset = (step // period * count) % max(1, live_slots.numel())
    slots = torch.roll(live_slots, shifts=-offset)[:count]
    old_ids = body.stable_id[slots].clone()
    next_id = int(body.stable_id.max().item()) + 1
    new_ids = torch.arange(next_id, next_id + count, dtype=torch.int64, device=body.stable_id.device)
    body.alive[slots] = False
    body.v_com[slots] = 0
    body.x_com[slots] = 0
    body.gait_time[slots] = 0
    body.stable_id[slots] = new_ids
    body.alive[slots] = True
    return LifecycleEvent(
        step=step,
        slots=tuple(int(x) for x in slots.cpu()),
        old_ids=tuple(int(x) for x in old_ids.cpu()),
        new_ids=tuple(int(x) for x in new_ids.cpu()),
    )


def tensor_addresses(body: BodyBatch) -> tuple[int, ...]:
    return tuple(
        tensor.data_ptr()
        for tensor in (
            body.alive,
            body.stable_id,
            body.seg_mask,
            body.local_pos,
            body.local_rot,
            body.abc,
            body.x_com,
            body.v_com,
        )
    )
