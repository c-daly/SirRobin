"""Ordered composition root for one exact ecological step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.contracts import EconomyStepLedger
from sirrobin.economy.ledger import MassLedger
from sirrobin.economy.reactions import reaction_step
from sirrobin.economy.state import EconomyState
from sirrobin.fields.transport import mix_vertical, sink_vertical


@dataclass(frozen=True, slots=True)
class EconomyAdvance:
    state: EconomyState
    ledger: EconomyStepLedger


def _advance_economy_in_place(
    state: EconomyState,
    config: EconomyConfig,
) -> EconomyStepLedger:
    before = state.total_per_world()
    reaction = reaction_step(state, config)
    sinking_q = torch.zeros_like(before)
    mixing_q = torch.zeros_like(before)
    transport_shortfall_q = torch.zeros_like(before)
    interventions = torch.zeros_like(before)
    sub_dt = config.dt_eco_s / config.transport_substeps
    for _ in range(config.transport_substeps):
        if config.sinking_speed_m_s > 0:
            sinking = sink_vertical(
                state.bd_q,
                state.carries.sinking_mol,
                config,
                dt_s=sub_dt,
            )
            sinking_q += sinking.moved_q
            transport_shortfall_q += sinking.shortfall_q
            interventions += sinking.intervention_count
        for reservoir, carry, diffusivity in (
            (state.nd_q, state.carries.mix_nd_mol, config.kz_nd_m2_s),
            (state.bp_q, state.carries.mix_bp_mol, config.kz_bp_m2_s),
            (state.bm_q, state.carries.mix_bm_mol, config.kz_bm_m2_s),
        ):
            if diffusivity > 0:
                mixed = mix_vertical(
                    reservoir,
                    carry,
                    diffusivity,
                    config,
                    dt_s=sub_dt,
                )
                mixing_q += mixed.moved_q
                transport_shortfall_q += mixed.shortfall_q
                interventions += mixed.intervention_count
    state.step.add_(1)
    state.time_s.add_(config.dt_eco_s)
    state.buffer_parity.bitwise_xor_(1)
    after = state.total_per_world()
    # This ledger proves that the field subsystem's own reaction/transport step
    # conserves what entered it. The composed world's persistent baseline also
    # includes creature stores and is enforced by the runtime composition.
    closed = MassLedger(before).close_books(state)
    return EconomyStepLedger(
        reaction.production_q,
        reaction.producer_maintenance_q,
        reaction.producer_mortality_q,
        reaction.decomposition_q,
        reaction.microbial_credit_q,
        reaction.dissolved_credit_q,
        reaction.microbial_turnover_q,
        sinking_q,
        mixing_q,
        reaction.shortfall_q,
        transport_shortfall_q,
        before,
        after,
        closed,
        interventions,
    )


def advance_economy_unchecked(
    state: EconomyState,
    config: EconomyConfig,
) -> EconomyAdvance:
    """Advance a boundary-validated field state without mutating the input."""

    candidate = state.clone()
    return EconomyAdvance(candidate, _advance_economy_in_place(candidate, config))


class EconomyKernel:
    """Validated stateful adapter retained for the reference engine."""

    def __init__(self, state: EconomyState, config: EconomyConfig):
        config.validate()
        state.validate(config)
        self.state = state
        self.config = config

    def step(self) -> EconomyStepLedger:
        return _advance_economy_in_place(self.state, self.config)
