"""Authoritative composition root for the headless world.

Tranche A seam. The ecological economy and the live mechanics are composed here and
share one declared schedule. No matter, energy, or intent crosses between them yet;
that coupling is Tranche C/D work and must arrive as explicit transfers through the
whole-world ledger, never as a side effect of stepping.

Authority follows the recovery synthesis section 3.2: one authoritative state per
quantity, with derived data allowed where it is a rebuildable function of that
authority AND one-way. The genotype is the authority; the developed body is a
rebuildable cache. `develop()` returns four of its fields by reference, so those are
cloned here — otherwise writing the cache would write the heredity authority, which
is exactly the write-back section 3.2 forbids.

Simulation time: `EconomyState.time_s` is the authoritative ecological clock and
`sim_time_s` is a read-only view of it. `LiveState.gait_time_s` is the mechanics
sub-clock, advanced by each canonical step or by explicitly verified periodic-step
coverage. The runner keeps the clocks in step on successful advances. A failed
post-step closure check leaves the world mutated and arrested rather than pretending
the advance was atomic.

Simulation time belongs to the core. Nothing here depends on a render frame and the
whole module runs without Unity.

Tracked creature structure and reserve are a separate integer authority from physical
body mass. The whole-world baseline sums them with the four field reservoirs. The
population feeding, maintenance, and paid exact-clone birth transactions cross that
seam; later lifecycle depth must use it without giving either subsystem a second global
ledger.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from sirrobin.core.live_world import advance_live_world, initialize_live_state
from sirrobin.core.material import (
    CreatureMaterialState,
    MaterialEnergyConfig,
    MatterTotals,
    WholeWorldMatterLedger,
    close_world_matter,
    matter_totals,
)
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.contracts import EconomyStepLedger
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel
from sirrobin.fields.geometry import GridGeometry
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample, LiveStepLedger
from sirrobin.physics.live_config import LiveLocomotionConfig

# `develop()` returns these by reference from the genotype. Cloning them keeps the
# developed body a one-way cache; without it, in-place writes to the body (the
# established lifecycle idiom, see benchmarks/lifecycle.py) rewrite the genotype.
_ALIASED_BODY_FIELDS = ("alive", "stable_id", "swim_freq_hz", "swim_wave_rad_per_depth")


class HeadlessWorld:
    """Owns every authoritative live quantity of the composed world."""

    def __init__(
        self,
        *,
        genotype: GenotypeBatch,
        fluid: FluidSample,
        live_config: LiveLocomotionConfig,
        economy_state: EconomyState,
        economy_config: EconomyConfig,
        creature_material_state: CreatureMaterialState,
        material_energy_config: MaterialEnergyConfig,
    ) -> None:
        live_config.validate()
        economy_config.validate()
        if genotype.alive.shape[0] != economy_config.worlds:
            raise ValueError(
                f"population spans {genotype.alive.shape[0]} worlds but the economy spans "
                f"{economy_config.worlds}"
            )

        self.genotype = genotype
        maximum_ids = genotype.stable_id.max(dim=1).values
        if bool((maximum_ids >= torch.iinfo(torch.int64).max).any()):
            raise ValueError("stable ID allocator is exhausted")
        self._next_stable_id = maximum_ids + 1
        body = develop(genotype)
        self.body = replace(
            body, **{name: getattr(body, name).clone() for name in _ALIASED_BODY_FIELDS}
        )
        self.live_state = initialize_live_state(self.body)
        self._require_matching_fluid(fluid)
        self.fluid = fluid
        self.live_config = live_config
        self.economy = EconomyKernel(economy_state, economy_config)
        self.geometry = GridGeometry.from_config(economy_config)
        self.creature_material = creature_material_state
        if not isinstance(material_energy_config, MaterialEnergyConfig):
            raise TypeError("material_energy_config must be MaterialEnergyConfig")
        self._material_energy_config = material_energy_config
        self.creature_material.validate(
            self.body.alive,
            q_mass_mol=economy_config.q_mass_mol,
            reserve_j_per_q=material_energy_config.reserve_j_per_q,
        )
        initial_matter = self.matter_totals()
        if not bool(initial_matter.raw_reservoirs_valid.all()):
            raise ValueError("whole-world inventory exceeds the configured safe reduction bound")
        self._expected_matter_total_q = initial_matter.total_q.clone()

    def _require_matching_fluid(self, fluid: FluidSample) -> None:
        """`FluidSample` carries no validate(); a mis-shaped one silently broadcasts.

        A (1,1) density against a (1,2) body gives creature 1 creature 0's water
        without error, which changes the physics and nothing detects it.
        """
        lead = tuple(self.body.alive.shape)
        checks = (
            ("density_kg_m3", fluid.density_kg_m3, lead),
            ("velocity_enu_m_s", fluid.velocity_enu_m_s, (*lead, 3)),
        )
        for name, tensor, expected in checks:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"fluid {name} must have shape {expected}, got {tuple(tensor.shape)}")
            if tensor.dtype != self.body.mass_sim.dtype:
                raise ValueError(f"fluid {name} dtype must match the body's {self.body.mass_sim.dtype}")
            if tensor.device != self.body.alive.device:
                raise ValueError(f"fluid {name} device must match the body's {self.body.alive.device}")

    @property
    def economy_state(self) -> EconomyState:
        return self.economy.state

    @property
    def economy_config(self) -> EconomyConfig:
        return self.economy.config

    @property
    def sim_time_s(self) -> float:
        """Read-only view of the authoritative ecological clock."""
        return float(self.economy.state.time_s)

    @property
    def material_energy_config(self) -> MaterialEnergyConfig:
        """Immutable world binding for producer and reserve chemical energy."""
        return self._material_energy_config

    @property
    def next_stable_id(self) -> torch.Tensor:
        """Read-only copy of the per-world monotonic ID allocator state."""
        return self._next_stable_id.clone()

    def _allocate_stable_id(self, world_index: int) -> int:
        """Consume one ID after a lifecycle transaction has passed all preflight."""
        value = int(self._next_stable_id[world_index].item())
        if value >= torch.iinfo(torch.int64).max:
            raise ValueError("stable ID allocator is exhausted")
        self._next_stable_id[world_index] = value + 1
        return value

    def rebuild_body(self) -> None:
        """Regenerate the developed-body cache from the genotype authority."""
        body = develop(self.genotype)
        self.body = replace(
            body, **{name: getattr(body, name).clone() for name in _ALIASED_BODY_FIELDS}
        )
        self.creature_material.validate(
            self.body.alive,
            q_mass_mol=self.economy_config.q_mass_mol,
            reserve_j_per_q=self.material_energy_config.reserve_j_per_q,
        )

    def matter_totals(self) -> MatterTotals:
        """Read-only exact census of field and creature nutrient reservoirs."""
        return matter_totals(
            self.economy_state,
            self.creature_material,
            alive=self.body.alive,
            field_shape=self.economy_config.shape,
            max_inventory_q=self.economy_config.max_inventory_q,
            q_mass_mol=self.economy_config.q_mass_mol,
            reserve_j_per_q=self.material_energy_config.reserve_j_per_q,
        )

    @property
    def expected_matter_total_q(self) -> torch.Tensor:
        """Copy of the immutable per-world nutrient baseline."""
        return self._expected_matter_total_q.clone()

    def close_matter_step(self, before: MatterTotals) -> WholeWorldMatterLedger:
        """Close the authoritative whole-world nutrient ledger after one tick."""
        return close_world_matter(
            expected_total_q=self._expected_matter_total_q,
            before=before,
            after=self.matter_totals(),
        )

    def _step_mechanics(self) -> LiveStepLedger:
        """Advance live mechanics by one frozen locomotion dt. Driven by the runner."""
        return advance_live_world(
            self.body,
            self.live_state,
            self.fluid,
            self.live_config,
            self.geometry,
        )

    def _step_economy(self) -> EconomyStepLedger:
        """Advance the ecological kernel, and with it the authoritative clock."""
        return self.economy.step()
