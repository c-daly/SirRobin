"""Closed-loop dynamics corpus and timestep-convergence gates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel


@dataclass(frozen=True, slots=True)
class DynamicsMetrics:
    initial_bp_q: int
    peak_bp_q: int
    minimum_nd_q: int
    final_bp_q: int
    late_mean_bp_q: float
    late_relative_range: float
    peak_time_s: float
    crash_time_s: float
    integrated_production_q: int
    integrated_decomposition_q: int
    integrated_microbe_turnover_q: int
    bd_ever_nonzero: bool
    bm_ever_nonzero: bool
    nd_recovered: bool
    books_closed: bool
    intervention_count: int
    transport_shortfall_q: int

    @property
    def bloom_passes(self) -> bool:
        return (
            self.peak_bp_q > self.initial_bp_q
            and self.peak_time_s < self.crash_time_s
            and self.final_bp_q < 0.70 * self.peak_bp_q
            and self.nd_recovered
            and self.bd_ever_nonzero
            and self.bm_ever_nonzero
            and self.late_mean_bp_q > 0
            and self.late_relative_range <= 0.25
            and self.integrated_decomposition_q > 0
            and self.integrated_microbe_turnover_q > 0
            and self.books_closed
            and self.intervention_count == 0
            and self.transport_shortfall_q == 0
        )


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    relative_differences: dict[str, float]
    peak_time_difference_s: float
    crash_time_difference_s: float
    passes: bool


def pulse_state(config: EconomyConfig, *, device: torch.device | str = "cpu") -> EconomyState:
    state = EconomyState.zeros(config, device=device)
    depth = (torch.arange(config.gz, dtype=torch.float64, device=device) + 0.5) * config.dz_m
    nd = torch.where(depth < 40.0, 8.0e-4, 2.0e-3)
    bp = torch.full_like(depth, 2.0e-5)
    nd_q = torch.round(nd * config.cell_volume_m3 / config.q_mass_mol).to(torch.int64)
    bp_q = torch.round(bp * config.cell_volume_m3 / config.q_mass_mol).to(torch.int64)
    state.nd_q.copy_(nd_q.view(1, 1, 1, -1).expand(config.shape))
    state.bp_q.copy_(bp_q.view(1, 1, 1, -1).expand(config.shape))
    return state


def run_pulse(
    config: EconomyConfig,
    *,
    steps: int,
    force_d_dd_zero: bool = True,
    device: torch.device | str = "cpu",
) -> DynamicsMetrics:
    if steps < 2:
        raise ValueError("pulse requires at least two steps")
    run_config = replace(config, density_mortality_m3_mol_s=0.0) if force_d_dd_zero else config
    state = pulse_state(run_config, device=device)
    kernel = EconomyKernel(state, run_config)
    bp_trace = [int(state.bp_q.sum().item())]
    nd_trace = [int(state.nd_q.sum().item())]
    bd_nonzero = False
    bm_nonzero = False
    books_closed = True
    interventions = 0
    shortfalls = 0
    production = 0
    decomposition = 0
    microbe_turnover = 0
    for _ in range(steps):
        ledger = kernel.step()
        bp_trace.append(int(state.bp_q.sum().item()))
        nd_trace.append(int(state.nd_q.sum().item()))
        bd_nonzero |= bool(state.bd_q.count_nonzero())
        bm_nonzero |= bool(state.bm_q.count_nonzero())
        books_closed &= bool(ledger.books_closed.all())
        interventions += int(ledger.intervention_count.sum().item())
        shortfalls += int(ledger.transport_shortfall_q.sum().item())
        production += int(ledger.production_q.sum().item())
        decomposition += int(ledger.decomposition_q.sum().item())
        microbe_turnover += int(ledger.microbial_turnover_q.sum().item())
    peak_index = max(range(len(bp_trace)), key=bp_trace.__getitem__)
    peak = bp_trace[peak_index]
    crash_index = next(
        (index for index in range(peak_index + 1, len(bp_trace)) if bp_trace[index] < 0.70 * peak),
        len(bp_trace) - 1,
    )
    late_count = max(10, len(bp_trace) // 10)
    late = bp_trace[-late_count:]
    min_nd_index = min(range(len(nd_trace)), key=nd_trace.__getitem__)
    return DynamicsMetrics(
        initial_bp_q=bp_trace[0],
        peak_bp_q=peak,
        minimum_nd_q=nd_trace[min_nd_index],
        final_bp_q=bp_trace[-1],
        late_mean_bp_q=sum(late) / late_count,
        late_relative_range=(max(late) - min(late)) / max(sum(late) / late_count, 1.0),
        peak_time_s=peak_index * run_config.dt_eco_s,
        crash_time_s=crash_index * run_config.dt_eco_s,
        integrated_production_q=production,
        integrated_decomposition_q=decomposition,
        integrated_microbe_turnover_q=microbe_turnover,
        bd_ever_nonzero=bd_nonzero,
        bm_ever_nonzero=bm_nonzero,
        nd_recovered=nd_trace[-1] > nd_trace[min_nd_index],
        books_closed=books_closed,
        intervention_count=interventions,
        transport_shortfall_q=shortfalls,
    )


def compare_half_timestep(
    coarse: DynamicsMetrics, fine: DynamicsMetrics, coarse_dt_s: float
) -> ConvergenceResult:
    pairs = {
        "peak_bp_q": (coarse.peak_bp_q, fine.peak_bp_q),
        "minimum_nd_q": (coarse.minimum_nd_q, fine.minimum_nd_q),
        "late_mean_bp_q": (coarse.late_mean_bp_q, fine.late_mean_bp_q),
        "integrated_production_q": (
            coarse.integrated_production_q,
            fine.integrated_production_q,
        ),
        "integrated_decomposition_q": (
            coarse.integrated_decomposition_q,
            fine.integrated_decomposition_q,
        ),
        "integrated_microbe_turnover_q": (
            coarse.integrated_microbe_turnover_q,
            fine.integrated_microbe_turnover_q,
        ),
    }
    differences = {
        name: abs(left - right) / max(abs(left), abs(right), 1.0) for name, (left, right) in pairs.items()
    }
    peak_time = abs(coarse.peak_time_s - fine.peak_time_s)
    crash_time = abs(coarse.crash_time_s - fine.crash_time_s)
    passes = (
        coarse.bloom_passes
        and fine.bloom_passes
        and all(value <= 0.05 for value in differences.values())
        and peak_time <= coarse_dt_s
        and crash_time <= coarse_dt_s
    )
    return ConvergenceResult(differences, peak_time, crash_time, passes)
