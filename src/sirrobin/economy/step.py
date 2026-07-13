"""Ordered composition root for one exact ecological step."""

from __future__ import annotations

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.contracts import EconomyStepLedger
from sirrobin.economy.ledger import MassLedger
from sirrobin.economy.reactions import reaction_step
from sirrobin.economy.state import EconomyState
from sirrobin.fields.transport import mix_vertical, sink_vertical


class EconomyKernel:
    def __init__(self, state: EconomyState, config: EconomyConfig):
        config.validate()
        state.validate(config)
        self.state = state
        self.config = config
        self.mass_ledger = MassLedger.from_state(state)

    def step(self) -> EconomyStepLedger:
        state, config = self.state, self.config
        before = state.total_per_world()
        reaction = reaction_step(state, config)
        sinking_q = torch.zeros_like(before)
        mixing_q = torch.zeros_like(before)
        transport_shortfall_q = torch.zeros_like(before)
        interventions = torch.zeros_like(before)
        sub_dt = config.dt_eco_s / config.transport_substeps
        for _ in range(config.transport_substeps):
            if config.sinking_speed_m_s > 0:
                sinking = sink_vertical(state.bd_q, state.carries.sinking_mol, config, dt_s=sub_dt)
                sinking_q += sinking.moved_q
                transport_shortfall_q += sinking.shortfall_q
                interventions += sinking.intervention_count
            for reservoir, carry, diffusivity in (
                (
                    state.nd_q,
                    state.carries.mix_nd_mol,
                    config.kz_nd_m2_s,
                ),
                (
                    state.bp_q,
                    state.carries.mix_bp_mol,
                    config.kz_bp_m2_s,
                ),
                (
                    state.bm_q,
                    state.carries.mix_bm_mol,
                    config.kz_bm_m2_s,
                ),
            ):
                if diffusivity > 0:
                    mixed = mix_vertical(reservoir, carry, diffusivity, config, dt_s=sub_dt)
                    mixing_q += mixed.moved_q
                    transport_shortfall_q += mixed.shortfall_q
                    interventions += mixed.intervention_count
        state.step.add_(1)
        state.time_s.add_(config.dt_eco_s)
        state.buffer_parity.bitwise_xor_(1)
        after = state.total_per_world()
        closed = self.mass_ledger.close_books(state)
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
