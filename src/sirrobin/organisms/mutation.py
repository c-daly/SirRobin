"""Deterministic batched parametric mutation for committed paid births."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from sirrobin.genetics.develop import develop_unchecked
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.numerics.quat import multiply, normalize
from sirrobin.organisms.lifecycle import LifecycleLedger
from sirrobin.organisms.random import identity_uniform, identity_word_u31
from sirrobin.organisms.state import PopulationState

JOINT_AMPLITUDE = 1
SWIM_FREQUENCY = 2
SWIM_WAVE = 3
SEGMENT_RESHAPE = 4
ATTACHMENT_POSITION = 5
ATTACHMENT_ANGLE = 6
SEGMENT_BUD = 7
SEGMENT_VESTIGIAL = 8

_U31_PRIME = 2_147_483_647


@dataclass(frozen=True, slots=True)
class MutationConfig:
    seed: int
    joint_amplitude: bool = True
    swim_frequency: bool = True
    swim_wave: bool = True
    segment_reshape: bool = True
    attachment_position: bool = True
    attachment_angle: bool = True
    segment_bud: bool = True
    segment_vestigial: bool = True
    joint_amplitude_step_rad: float = math.radians(2.0)
    swim_frequency_step_hz: float = 0.1
    swim_wave_step_rad_per_depth: float = math.radians(2.0)
    segment_log_axis_step: float = math.log(1.05)
    attachment_position_step: float = 0.02
    attachment_angle_step_rad: float = math.radians(2.0)
    attachment_position_limit: float = 2.0
    bud_axis_fraction: float = 0.15
    vestigial_axis_fraction: float = 0.90
    vestigial_remove_axis_fraction: float = 0.05
    mutation_rate_per_locus: float = 0.002
    max_mutations_per_birth: int = 3
    parameter_event_weight: int = 10
    topology_event_weight: int = 1

    def validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("mutation seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("mutation seed must be in [0,2^63)")
        enabled = (
            self.joint_amplitude,
            self.swim_frequency,
            self.swim_wave,
            self.segment_reshape,
            self.attachment_position,
            self.attachment_angle,
            self.segment_bud,
            self.segment_vestigial,
        )
        if any(not isinstance(value, bool) for value in enabled):
            raise TypeError("mutation trait switches must be boolean")
        if not any(enabled):
            raise ValueError("at least one mutation trait must be enabled")
        steps = (
            self.joint_amplitude_step_rad,
            self.swim_frequency_step_hz,
            self.swim_wave_step_rad_per_depth,
            self.segment_log_axis_step,
            self.attachment_position_step,
            self.attachment_angle_step_rad,
            self.attachment_position_limit,
        )
        if any(
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(step)
            or step <= 0.0
            for step in steps
        ):
            raise ValueError("mutation steps must be finite and positive")
        fractions = (
            self.bud_axis_fraction,
            self.vestigial_axis_fraction,
            self.vestigial_remove_axis_fraction,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value < 1.0
            for value in fractions
        ):
            raise ValueError("developmental morphology fractions must be in (0,1)")
        if self.vestigial_remove_axis_fraction >= self.vestigial_axis_fraction:
            raise ValueError("removal scale must be smaller than the vestigial step")
        if (
            isinstance(self.mutation_rate_per_locus, bool)
            or not isinstance(self.mutation_rate_per_locus, (int, float))
            or not math.isfinite(self.mutation_rate_per_locus)
            or not 0.0 <= self.mutation_rate_per_locus <= 1.0
        ):
            raise ValueError("mutation_rate_per_locus must be in [0,1]")
        integers = (
            self.max_mutations_per_birth,
            self.parameter_event_weight,
            self.topology_event_weight,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integers
        ):
            raise ValueError("mutation event bounds and weights must be positive integers")


@dataclass(frozen=True, slots=True)
class MutationLedger:
    mutated: torch.Tensor
    trait_code: torch.Tensor
    locus: torch.Tensor
    component: torch.Tensor
    parent_value: torch.Tensor
    child_value: torch.Tensor
    unavailable: torch.Tensor
    mutation_count: torch.Tensor
    event_applied: torch.Tensor
    event_trait_code: torch.Tensor
    event_locus: torch.Tensor
    event_component: torch.Tensor
    event_parent_value: torch.Tensor
    event_child_value: torch.Tensor


@dataclass(frozen=True, slots=True)
class MutationStep:
    genotype: GenotypeBatch
    ledger: MutationLedger


def _gather_parent(values: torch.Tensor, parent_slot: torch.Tensor) -> torch.Tensor:
    tail = values.shape[2:]
    index = parent_slot[(...,) + (None,) * len(tail)].expand(
        *parent_slot.shape, *tail
    )
    return torch.gather(values, 1, index)


def _clone_committed_births(
    genotype: GenotypeBatch,
    population: PopulationState,
    lifecycle: LifecycleLedger,
) -> GenotypeBatch:
    born = lifecycle.born
    parent_slot = lifecycle.parent_slot_for_child.clamp_min(0)
    values: dict[str, torch.Tensor] = {}
    for field in fields(genotype):
        name = field.name
        current = getattr(genotype, name)
        if name == "alive":
            values[name] = population.alive
        elif name == "stable_id":
            values[name] = population.stable_id
        else:
            parent = _gather_parent(current, parent_slot)
            mask = born[(...,) + (None,) * (current.ndim - born.ndim)]
            values[name] = torch.where(mask, parent, current)
    return GenotypeBatch(**values)


def _bounded_step(
    parent: torch.Tensor,
    direction: torch.Tensor,
    *,
    step: float,
    lower: float,
    upper: float,
) -> torch.Tensor:
    proposed = (parent + direction * step).clamp(lower, upper)
    alternate = (parent - direction * step).clamp(lower, upper)
    return torch.where(proposed != parent, proposed, alternate)


def _clone_genotype(genotype: GenotypeBatch) -> GenotypeBatch:
    return GenotypeBatch(
        **{
            field.name: getattr(genotype, field.name).clone()
            for field in fields(genotype)
        }
    )


def _event_identity(
    stable_id: torch.Tensor,
    event_index: torch.Tensor,
) -> torch.Tensor:
    counter = torch.remainder(event_index.to(torch.int64), _U31_PRIME)
    if counter.ndim == 0:
        counter = counter.expand(stable_id.shape[0])
    return torch.remainder(
        stable_id + counter[:, None] * 1_000_003,
        _U31_PRIME,
    )


def _select_locus(
    mask: torch.Tensor,
    word: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=-1, dtype=torch.int64)
    target = torch.remainder(word, count.clamp_min(1))
    rank = mask.to(torch.int64).cumsum(dim=-1) - 1
    selected = mask & (rank == target[..., None])
    indices = torch.arange(mask.shape[-1], dtype=torch.int64, device=mask.device)
    locus = torch.where(selected, indices, torch.zeros_like(indices)).max(dim=-1).values
    return selected, locus


def _propose_single_offspring_mutation(
    genotype: GenotypeBatch,
    body,
    requested_birth: torch.Tensor,
    event_index: torch.Tensor,
    config: MutationConfig,
) -> MutationStep:
    """Propose one gradual hereditary change per requesting parent slot.

    Candidates remain indexed by their parent slots. A later lifecycle transaction
    can therefore develop and price every candidate before accepting a birth, then
    gather only accepted candidates into child slots. Randomness is keyed by parent
    identity and the authoritative interval counter, not by mutable slot order.
    """

    candidate = _clone_genotype(genotype)
    requested = requested_birth & genotype.alive
    worlds, capacity = requested.shape
    device = requested.device
    dtype = genotype.node_log_axes_flu_m.dtype
    world_index = torch.arange(worlds, dtype=torch.int64, device=device)[:, None].expand(
        worlds, capacity
    )
    identity = _event_identity(genotype.stable_id, event_index)

    node_slots = torch.arange(
        genotype.node_mask.shape[-1], dtype=torch.int64, device=device
    )
    edge_slots = torch.arange(
        genotype.edge_mask.shape[-1], dtype=torch.int64, device=device
    )
    active_node = genotype.node_mask & genotype.node_expressed
    active_edge = genotype.edge_mask
    joint_node = active_node & (node_slots > 0)

    source_matches = (
        body.source_node.to(torch.int64)[..., :, None] == node_slots
    ) & body.seg_mask[..., :, None]
    emission_count = source_matches.sum(dim=-2, dtype=torch.int64)
    unique_emission_source = active_node & (emission_count == 1)
    free_node = ~genotype.node_mask
    free_edge = ~genotype.edge_mask

    outgoing = (
        active_edge[..., :, None]
        & (genotype.edge_src.to(torch.int64)[..., :, None] == node_slots)
    ).any(dim=-2)
    incoming = (
        active_edge[..., :, None]
        & (genotype.edge_dst.to(torch.int64)[..., :, None] == node_slots)
    ).sum(dim=-2)
    leaf_node = active_node & (node_slots > 0) & ~outgoing & (incoming == 1)

    enabled = torch.tensor(
        (
            config.joint_amplitude,
            config.swim_frequency,
            config.swim_wave,
            config.segment_reshape,
            config.attachment_position,
            config.attachment_angle,
            config.segment_bud,
            config.segment_vestigial,
        ),
        dtype=torch.bool,
        device=device,
    )
    available = torch.stack(
        (
            enabled[0] & joint_node.any(dim=-1),
            enabled[1].expand_as(requested),
            enabled[2].expand_as(requested),
            enabled[3] & active_node.any(dim=-1),
            enabled[4] & active_edge.any(dim=-1),
            enabled[5] & active_edge.any(dim=-1),
            enabled[6]
            & free_node.any(dim=-1)
            & free_edge.any(dim=-1)
            & unique_emission_source.any(dim=-1)
            & (body.seg_mask.sum(dim=-1) < body.seg_mask.shape[-1] - 1)
            & (body.truncated_candidate_count == 0),
            enabled[7] & leaf_node.any(dim=-1),
        ),
        dim=-1,
    )
    trait_weights = torch.tensor(
        (
            config.parameter_event_weight,
            config.parameter_event_weight,
            config.parameter_event_weight,
            config.parameter_event_weight,
            config.parameter_event_weight,
            config.parameter_event_weight,
            config.topology_event_weight,
            config.topology_event_weight,
        ),
        dtype=torch.int64,
        device=device,
    )
    available_weights = available.to(torch.int64) * trait_weights
    available_weight = available_weights.sum(dim=-1, dtype=torch.int64)
    trait_word = identity_word_u31(identity, world_index, seed=config.seed, stream=1)
    target_weight = torch.remainder(trait_word, available_weight.clamp_min(1))
    upper_weight = available_weights.cumsum(dim=-1)
    lower_weight = upper_weight - available_weights
    selected_trait = available & (target_weight[..., None] >= lower_weight) & (
        target_weight[..., None] < upper_weight
    )
    trait_codes = torch.tensor(
        (
            JOINT_AMPLITUDE,
            SWIM_FREQUENCY,
            SWIM_WAVE,
            SEGMENT_RESHAPE,
            ATTACHMENT_POSITION,
            ATTACHMENT_ANGLE,
            SEGMENT_BUD,
            SEGMENT_VESTIGIAL,
        ),
        dtype=torch.int64,
        device=device,
    )
    trait_code = (selected_trait.to(torch.int64) * trait_codes).sum(dim=-1)
    trait_code = torch.where(requested, trait_code, 0)
    unavailable = requested & (available_weight == 0)

    direction_word = identity_word_u31(
        identity, world_index, seed=config.seed, stream=2
    )
    direction = torch.where(
        torch.remainder(direction_word, 2) == 1,
        torch.ones_like(genotype.swim_freq_hz),
        -torch.ones_like(genotype.swim_freq_hz),
    )
    locus_word = identity_word_u31(
        identity, world_index, seed=config.seed, stream=3
    )
    component_word = identity_word_u31(
        identity, world_index, seed=config.seed, stream=4
    )
    component = torch.remainder(component_word, 3)

    selected_joint, joint_locus = _select_locus(joint_node, locus_word)
    selected_shape, shape_locus = _select_locus(active_node, locus_word)
    selected_edge, edge_locus = _select_locus(active_edge, locus_word)
    selected_leaf, leaf_locus = _select_locus(leaf_node, locus_word)
    selected_source, source_locus = _select_locus(unique_emission_source, locus_word)

    joint_parent = torch.gather(
        candidate.node_joint_amp_rad, -1, joint_locus[..., None]
    ).squeeze(-1)
    joint_child = _bounded_step(
        joint_parent,
        direction,
        step=config.joint_amplitude_step_rad,
        lower=0.0,
        upper=0.5 * math.pi,
    )
    joint_update = requested & (trait_code == JOINT_AMPLITUDE)
    candidate.node_joint_amp_rad = torch.where(
        joint_update[..., None] & selected_joint,
        joint_child[..., None],
        candidate.node_joint_amp_rad,
    )

    frequency_parent = candidate.swim_freq_hz
    frequency_child = _bounded_step(
        frequency_parent,
        direction,
        step=config.swim_frequency_step_hz,
        lower=0.0,
        upper=10.0,
    )
    frequency_update = requested & (trait_code == SWIM_FREQUENCY)
    candidate.swim_freq_hz = torch.where(
        frequency_update, frequency_child, candidate.swim_freq_hz
    )

    wave_parent = candidate.swim_wave_rad_per_depth
    wave_child = _bounded_step(
        wave_parent,
        direction,
        step=config.swim_wave_step_rad_per_depth,
        lower=-math.pi,
        upper=math.pi,
    )
    wave_update = requested & (trait_code == SWIM_WAVE)
    candidate.swim_wave_rad_per_depth = torch.where(
        wave_update, wave_child, candidate.swim_wave_rad_per_depth
    )

    shape_component = component[..., None] == torch.arange(3, device=device)
    selected_shape_component = selected_shape[..., :, None] & shape_component[..., None, :]
    reshape_update = requested & (trait_code == SEGMENT_RESHAPE)
    shape_before_log = torch.gather(
        genotype.node_log_axes_flu_m,
        -2,
        shape_locus[..., None, None].expand(*shape_locus.shape, 1, 3),
    ).squeeze(-2)
    shape_parent_log = torch.gather(
        shape_before_log, -1, component[..., None]
    ).squeeze(-1)
    shape_child_log = shape_parent_log + direction * config.segment_log_axis_step
    candidate.node_log_axes_flu_m = torch.where(
        reshape_update[..., None, None] & selected_shape_component,
        shape_child_log[..., None, None],
        candidate.node_log_axes_flu_m,
    )

    edge_before_attach = torch.gather(
        genotype.edge_attach_parent_axes,
        -2,
        edge_locus[..., None, None].expand(*edge_locus.shape, 1, 3),
    ).squeeze(-2)
    attach_parent = torch.gather(
        edge_before_attach, -1, component[..., None]
    ).squeeze(-1)
    attach_child = _bounded_step(
        attach_parent,
        direction,
        step=config.attachment_position_step,
        lower=-config.attachment_position_limit,
        upper=config.attachment_position_limit,
    )
    attach_update = requested & (trait_code == ATTACHMENT_POSITION)
    selected_edge_component = selected_edge[..., :, None] & shape_component[..., None, :]
    candidate.edge_attach_parent_axes = torch.where(
        attach_update[..., None, None] & selected_edge_component,
        attach_child[..., None, None],
        candidate.edge_attach_parent_axes,
    )

    angle_update = requested & (trait_code == ATTACHMENT_ANGLE)
    angle = direction * config.attachment_angle_step_rad
    axis = torch.nn.functional.one_hot(component, num_classes=3).to(dtype)
    delta_rotation = torch.cat(
        (
            axis * torch.sin(0.5 * angle)[..., None],
            torch.cos(0.5 * angle)[..., None],
        ),
        dim=-1,
    )
    edge_before_rotation = torch.gather(
        genotype.edge_rot_flu,
        -2,
        edge_locus[..., None, None].expand(*edge_locus.shape, 1, 4),
    ).squeeze(-2)
    edge_after_rotation = normalize(multiply(edge_before_rotation, delta_rotation))
    candidate.edge_rot_flu = torch.where(
        angle_update[..., None, None] & selected_edge[..., :, None],
        edge_after_rotation[..., None, :],
        candidate.edge_rot_flu,
    )

    bud_update = requested & (trait_code == SEGMENT_BUD)
    new_node = torch.argmax(free_node.to(torch.int64), dim=-1)
    new_edge = torch.argmax(free_edge.to(torch.int64), dim=-1)
    new_node_selected = node_slots == new_node[..., None]
    new_edge_selected = edge_slots == new_edge[..., None]
    source_axes = torch.gather(
        genotype.node_log_axes_flu_m,
        -2,
        source_locus[..., None, None].expand(*source_locus.shape, 1, 3),
    ).squeeze(-2)
    source_density = torch.gather(
        genotype.node_density_gene, -1, source_locus[..., None]
    ).squeeze(-1)
    next_iid = torch.maximum(
        genotype.node_iid.max(dim=-1).values,
        genotype.edge_iid.max(dim=-1).values,
    ) + 1
    candidate.node_mask = candidate.node_mask | (
        bud_update[..., None] & new_node_selected
    )
    candidate.node_iid = torch.where(
        bud_update[..., None] & new_node_selected,
        next_iid[..., None],
        candidate.node_iid,
    )
    candidate.node_type = torch.where(
        bud_update[..., None] & new_node_selected,
        torch.zeros_like(candidate.node_type),
        candidate.node_type,
    )
    candidate.node_log_axes_flu_m = torch.where(
        bud_update[..., None, None] & new_node_selected[..., None],
        source_axes[..., None, :] + math.log(config.bud_axis_fraction),
        candidate.node_log_axes_flu_m,
    )
    candidate.node_density_gene = torch.where(
        bud_update[..., None] & new_node_selected,
        source_density[..., None],
        candidate.node_density_gene,
    )
    for name in ("node_intake", "node_sense"):
        value = getattr(candidate, name)
        setattr(
            candidate,
            name,
            torch.where(
                bud_update[..., None] & new_node_selected,
                torch.zeros_like(value),
                value,
            ),
        )
    candidate.node_expressed = candidate.node_expressed | (
        bud_update[..., None] & new_node_selected
    )
    candidate.node_joint_amp_rad = torch.where(
        bud_update[..., None] & new_node_selected,
        torch.zeros_like(candidate.node_joint_amp_rad),
        candidate.node_joint_amp_rad,
    )
    bud_hinge = torch.zeros_like(candidate.node_hinge_axis_flu)
    bud_hinge[..., 2] = 1.0
    candidate.node_hinge_axis_flu = torch.where(
        bud_update[..., None, None] & new_node_selected[..., None],
        bud_hinge,
        candidate.node_hinge_axis_flu,
    )
    candidate.edge_mask = candidate.edge_mask | (
        bud_update[..., None] & new_edge_selected
    )
    candidate.edge_iid = torch.where(
        bud_update[..., None] & new_edge_selected,
        (next_iid + 1)[..., None],
        candidate.edge_iid,
    )
    candidate.edge_src = torch.where(
        bud_update[..., None] & new_edge_selected,
        source_locus.to(candidate.edge_src.dtype)[..., None],
        candidate.edge_src,
    )
    candidate.edge_dst = torch.where(
        bud_update[..., None] & new_edge_selected,
        new_node.to(candidate.edge_dst.dtype)[..., None],
        candidate.edge_dst,
    )
    bud_attach = torch.nn.functional.one_hot(component, num_classes=3).to(dtype)
    bud_attach = bud_attach * direction[..., None]
    candidate.edge_attach_parent_axes = torch.where(
        bud_update[..., None, None] & new_edge_selected[..., None],
        bud_attach[..., None, :],
        candidate.edge_attach_parent_axes,
    )
    bud_rotation = torch.zeros_like(candidate.edge_rot_flu)
    bud_rotation[..., 3] = 1.0
    candidate.edge_rot_flu = torch.where(
        bud_update[..., None, None] & new_edge_selected[..., None],
        bud_rotation,
        candidate.edge_rot_flu,
    )
    candidate.edge_scale = torch.where(
        bud_update[..., None] & new_edge_selected,
        torch.ones_like(candidate.edge_scale),
        candidate.edge_scale,
    )
    candidate.edge_mirror = torch.where(
        bud_update[..., None] & new_edge_selected,
        torch.zeros_like(candidate.edge_mirror),
        candidate.edge_mirror,
    )
    candidate.edge_recursion = torch.where(
        bud_update[..., None] & new_edge_selected,
        torch.ones_like(candidate.edge_recursion),
        candidate.edge_recursion,
    )

    vestigial_update = requested & (trait_code == SEGMENT_VESTIGIAL)
    leaf_log_axes = torch.gather(
        genotype.node_log_axes_flu_m,
        -2,
        leaf_locus[..., None, None].expand(*leaf_locus.shape, 1, 3),
    ).squeeze(-2)
    root_log_axes = genotype.node_log_axes_flu_m[..., 0, :]
    remove_leaf = vestigial_update & (
        (leaf_log_axes - root_log_axes)
        <= math.log(config.vestigial_remove_axis_fraction)
    ).all(dim=-1)
    shrink_leaf = vestigial_update & ~remove_leaf
    candidate.node_log_axes_flu_m = torch.where(
        shrink_leaf[..., None, None] & selected_leaf[..., :, None],
        candidate.node_log_axes_flu_m + math.log(config.vestigial_axis_fraction),
        candidate.node_log_axes_flu_m,
    )
    candidate.node_mask = candidate.node_mask & ~(
        remove_leaf[..., None] & selected_leaf
    )
    candidate.node_expressed = candidate.node_expressed & ~(
        remove_leaf[..., None] & selected_leaf
    )
    removed_edge = active_edge & (
        genotype.edge_dst.to(torch.int64) == leaf_locus[..., None]
    )
    candidate.edge_mask = candidate.edge_mask & ~(
        remove_leaf[..., None] & removed_edge
    )

    shape_parent = torch.exp(shape_parent_log)
    shape_child = torch.exp(shape_child_log)
    leaf_parent = torch.exp(leaf_log_axes.mean(dim=-1))
    leaf_child = torch.where(
        remove_leaf,
        torch.zeros_like(leaf_parent),
        leaf_parent * config.vestigial_axis_fraction,
    )
    bud_child = torch.exp(source_axes.mean(dim=-1)) * config.bud_axis_fraction
    parent_value = torch.where(
        joint_update,
        joint_parent,
        torch.where(
            frequency_update,
            frequency_parent,
            torch.where(
                wave_update,
                wave_parent,
                torch.where(
                    reshape_update,
                    shape_parent,
                    torch.where(
                        attach_update,
                        attach_parent,
                        torch.where(
                            angle_update,
                            torch.zeros_like(angle),
                            torch.where(
                                bud_update,
                                torch.zeros_like(bud_child),
                                leaf_parent,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    child_value = torch.where(
        joint_update,
        joint_child,
        torch.where(
            frequency_update,
            frequency_child,
            torch.where(
                wave_update,
                wave_child,
                torch.where(
                    reshape_update,
                    shape_child,
                    torch.where(
                        attach_update,
                        attach_child,
                        torch.where(
                            angle_update,
                            angle,
                            torch.where(bud_update, bud_child, leaf_child),
                        ),
                    ),
                ),
            ),
        ),
    )
    locus = torch.where(
        joint_update,
        joint_locus,
        torch.where(
            reshape_update,
            shape_locus,
            torch.where(
                attach_update | angle_update,
                edge_locus,
                torch.where(bud_update, new_node, leaf_locus),
            ),
        ),
    )
    uses_component = reshape_update | attach_update | angle_update
    mutated = requested & ~unavailable
    return MutationStep(
        candidate,
        MutationLedger(
            mutated=mutated,
            trait_code=trait_code,
            locus=torch.where(mutated, locus, torch.full_like(locus, -1)),
            component=torch.where(
                uses_component,
                component,
                torch.full_like(component, -1),
            ),
            parent_value=torch.where(mutated, parent_value, 0.0),
            child_value=torch.where(mutated, child_value, 0.0),
            unavailable=unavailable,
            mutation_count=mutated.to(torch.int64),
            event_applied=mutated[..., None],
            event_trait_code=trait_code[..., None],
            event_locus=torch.where(
                mutated, locus, torch.full_like(locus, -1)
            )[..., None],
            event_component=torch.where(
                uses_component,
                component,
                torch.full_like(component, -1),
            )[..., None],
            event_parent_value=torch.where(mutated, parent_value, 0.0)[..., None],
            event_child_value=torch.where(mutated, child_value, 0.0)[..., None],
        ),
    )


def propose_offspring_mutations(
    genotype: GenotypeBatch,
    body,
    requested_birth: torch.Tensor,
    event_index: torch.Tensor,
    config: MutationConfig,
) -> MutationStep:
    """Propose zero to a bounded number of mutation events per offspring.

    Event opportunity scales with the number of enabled mutable loci. Fixed event
    slots keep the kernel shape static for GPU compilation while allowing the
    biologically important outcomes of no mutation and multiple mutations.
    """

    candidate = genotype
    current_body = body
    requested = requested_birth & genotype.alive
    worlds, capacity = requested.shape
    device = requested.device
    world_index = torch.arange(worlds, dtype=torch.int64, device=device)[:, None].expand(
        worlds, capacity
    )
    primary_set = torch.zeros_like(requested)
    primary_trait = torch.zeros_like(genotype.stable_id)
    primary_locus = torch.full_like(genotype.stable_id, -1)
    primary_component = torch.full_like(genotype.stable_id, -1)
    primary_parent = torch.zeros_like(genotype.swim_freq_hz)
    primary_child = torch.zeros_like(genotype.swim_freq_hz)
    unavailable = torch.zeros_like(requested)
    applied_events = []
    trait_events = []
    locus_events = []
    component_events = []
    parent_events = []
    child_events = []

    for event_slot in range(config.max_mutations_per_birth):
        active_nodes = candidate.node_mask & candidate.node_expressed
        active_edges = candidate.edge_mask
        mutable_loci = torch.zeros_like(genotype.stable_id)
        if config.joint_amplitude:
            mutable_loci = mutable_loci + (
                active_nodes
                & (
                    torch.arange(active_nodes.shape[-1], device=device)
                    > 0
                )
            ).sum(dim=-1, dtype=torch.int64)
        if config.swim_frequency:
            mutable_loci = mutable_loci + 1
        if config.swim_wave:
            mutable_loci = mutable_loci + 1
        if config.segment_reshape:
            mutable_loci = mutable_loci + 3 * active_nodes.sum(
                dim=-1, dtype=torch.int64
            )
        if config.attachment_position:
            mutable_loci = mutable_loci + 3 * active_edges.sum(
                dim=-1, dtype=torch.int64
            )
        if config.attachment_angle:
            mutable_loci = mutable_loci + 3 * active_edges.sum(
                dim=-1, dtype=torch.int64
            )
        if config.segment_bud:
            mutable_loci = mutable_loci + 1
        if config.segment_vestigial:
            mutable_loci = mutable_loci + active_nodes.sum(
                dim=-1, dtype=torch.int64
            ).clamp_min(1) - 1

        event_probability = (
            mutable_loci.to(torch.float64)
            * config.mutation_rate_per_locus
            / config.max_mutations_per_birth
        ).clamp(0.0, 1.0)
        counter = event_index * config.max_mutations_per_birth + event_slot
        occurrence_identity = _event_identity(genotype.stable_id, counter)
        occurrence = identity_uniform(
            occurrence_identity,
            world_index,
            seed=config.seed,
            stream=20 + event_slot,
        ) < event_probability
        event_requested = requested & occurrence
        single = _propose_single_offspring_mutation(
            candidate,
            current_body,
            event_requested,
            counter,
            config,
        )
        candidate = single.genotype
        event_ledger = single.ledger
        take_primary = event_ledger.mutated & ~primary_set
        primary_trait = torch.where(
            take_primary, event_ledger.trait_code, primary_trait
        )
        primary_locus = torch.where(
            take_primary, event_ledger.locus, primary_locus
        )
        primary_component = torch.where(
            take_primary, event_ledger.component, primary_component
        )
        primary_parent = torch.where(
            take_primary, event_ledger.parent_value, primary_parent
        )
        primary_child = torch.where(
            take_primary, event_ledger.child_value, primary_child
        )
        primary_set |= event_ledger.mutated
        unavailable |= event_ledger.unavailable
        applied_events.append(event_ledger.mutated)
        trait_events.append(event_ledger.trait_code)
        locus_events.append(event_ledger.locus)
        component_events.append(event_ledger.component)
        parent_events.append(event_ledger.parent_value)
        child_events.append(event_ledger.child_value)
        if event_slot + 1 < config.max_mutations_per_birth:
            current_body = develop_unchecked(candidate)

    event_applied = torch.stack(applied_events, dim=-1)
    event_trait_code = torch.stack(trait_events, dim=-1)
    event_locus = torch.stack(locus_events, dim=-1)
    event_component = torch.stack(component_events, dim=-1)
    event_parent_value = torch.stack(parent_events, dim=-1)
    event_child_value = torch.stack(child_events, dim=-1)
    mutation_count = event_applied.sum(dim=-1, dtype=torch.int64)
    return MutationStep(
        candidate,
        MutationLedger(
            mutated=mutation_count > 0,
            trait_code=primary_trait,
            locus=primary_locus,
            component=primary_component,
            parent_value=primary_parent,
            child_value=primary_child,
            unavailable=unavailable,
            mutation_count=mutation_count,
            event_applied=event_applied,
            event_trait_code=event_trait_code,
            event_locus=event_locus,
            event_component=event_component,
            event_parent_value=event_parent_value,
            event_child_value=event_child_value,
        ),
    )


def commit_offspring_mutations(
    genotype: GenotypeBatch,
    proposal: MutationStep,
    population: PopulationState,
    lifecycle: LifecycleLedger,
) -> MutationStep:
    """Gather accepted parent-indexed candidates into assigned child slots."""

    born = lifecycle.born
    parent_slot = lifecycle.parent_slot_for_child.clamp_min(0)
    values: dict[str, torch.Tensor] = {}
    for field in fields(genotype):
        name = field.name
        current = getattr(genotype, name)
        if name == "alive":
            values[name] = population.alive
        elif name == "stable_id":
            values[name] = population.stable_id
        else:
            parent_candidate = _gather_parent(
                getattr(proposal.genotype, name), parent_slot
            )
            mask = born[(...,) + (None,) * (current.ndim - born.ndim)]
            values[name] = torch.where(mask, parent_candidate, current)

    def relocate(value: torch.Tensor, default: int | float | bool) -> torch.Tensor:
        gathered = _gather_parent(value, parent_slot)
        mask = born[(...,) + (None,) * (value.ndim - born.ndim)]
        return torch.where(mask, gathered, torch.full_like(value, default))

    source = proposal.ledger
    ledger = MutationLedger(
        mutated=relocate(source.mutated, False),
        trait_code=relocate(source.trait_code, 0),
        locus=relocate(source.locus, -1),
        component=relocate(source.component, -1),
        parent_value=relocate(source.parent_value, 0.0),
        child_value=relocate(source.child_value, 0.0),
        unavailable=relocate(source.unavailable, False),
        mutation_count=relocate(source.mutation_count, 0),
        event_applied=relocate(source.event_applied, False),
        event_trait_code=relocate(source.event_trait_code, 0),
        event_locus=relocate(source.event_locus, -1),
        event_component=relocate(source.event_component, -1),
        event_parent_value=relocate(source.event_parent_value, 0.0),
        event_child_value=relocate(source.event_child_value, 0.0),
    )
    return MutationStep(GenotypeBatch(**values), ledger)


def mutate_committed_births(
    genotype: GenotypeBatch,
    population: PopulationState,
    lifecycle: LifecycleLedger,
    config: MutationConfig,
) -> MutationStep:
    """Clone every committed child and mutate exactly one available scalar locus.

    The lifecycle transaction has already paid and assigned these births. This
    function contains no slot allocation, material transfer, or host random state.
    Invalid trait availability is reported as a device status flag so the enclosing
    chunk can arrest rather than publishing a partially specified newborn.
    """

    candidate = _clone_committed_births(genotype, population, lifecycle)
    born = lifecycle.born
    worlds, capacity = born.shape
    world_index = torch.arange(
        worlds, dtype=torch.int64, device=born.device
    )[:, None].expand(worlds, capacity)
    child_id = population.stable_id

    node_index = torch.arange(
        candidate.node_mask.shape[-1], dtype=torch.int64, device=born.device
    )
    joint_node = (
        candidate.node_mask
        & candidate.node_expressed
        & (node_index > 0)
    )
    joint_count = joint_node.sum(dim=-1, dtype=torch.int64)
    enabled = torch.tensor(
        (config.joint_amplitude, config.swim_frequency, config.swim_wave),
        dtype=torch.bool,
        device=born.device,
    )
    available = torch.stack(
        (
            enabled[0] & (joint_count > 0),
            enabled[1].expand_as(born),
            enabled[2].expand_as(born),
        ),
        dim=-1,
    )
    available_count = available.sum(dim=-1, dtype=torch.int64)
    trait_word = identity_word_u31(
        child_id,
        world_index,
        seed=config.seed,
        stream=1,
    )
    target_rank = torch.remainder(trait_word, available_count.clamp_min(1))
    trait_rank = available.to(torch.int64).cumsum(dim=-1) - 1
    selected_trait = available & (trait_rank == target_rank[..., None])
    trait_codes = torch.tensor(
        (JOINT_AMPLITUDE, SWIM_FREQUENCY, SWIM_WAVE),
        dtype=torch.int64,
        device=born.device,
    )
    trait_code = (selected_trait.to(torch.int64) * trait_codes).sum(dim=-1)
    trait_code = torch.where(born, trait_code, 0)
    unavailable = born & (available_count == 0)

    direction_word = identity_word_u31(
        child_id,
        world_index,
        seed=config.seed,
        stream=2,
    )
    direction = torch.where(
        torch.remainder(direction_word, 2) == 1,
        1.0,
        -1.0,
    ).to(candidate.swim_freq_hz.dtype)
    locus_word = identity_word_u31(
        child_id,
        world_index,
        seed=config.seed,
        stream=3,
    )
    locus_rank = torch.remainder(locus_word, joint_count.clamp_min(1))
    node_rank = joint_node.to(torch.int64).cumsum(dim=-1) - 1
    selected_node = joint_node & (node_rank == locus_rank[..., None])
    locus = torch.where(
        selected_node,
        node_index,
        torch.zeros_like(node_index),
    ).max(dim=-1).values

    joint_parent = torch.gather(
        candidate.node_joint_amp_rad, -1, locus[..., None]
    ).squeeze(-1)
    joint_child = _bounded_step(
        joint_parent,
        direction,
        step=config.joint_amplitude_step_rad,
        lower=0.0,
        upper=0.5 * math.pi,
    )
    joint_update = born & (trait_code == JOINT_AMPLITUDE)
    candidate.node_joint_amp_rad = torch.where(
        joint_update[..., None] & (node_index == locus[..., None]),
        joint_child[..., None],
        candidate.node_joint_amp_rad,
    )

    frequency_parent = candidate.swim_freq_hz
    frequency_child = _bounded_step(
        frequency_parent,
        direction,
        step=config.swim_frequency_step_hz,
        lower=0.0,
        upper=10.0,
    )
    frequency_update = born & (trait_code == SWIM_FREQUENCY)
    candidate.swim_freq_hz = torch.where(
        frequency_update, frequency_child, candidate.swim_freq_hz
    )

    wave_parent = candidate.swim_wave_rad_per_depth
    wave_child = _bounded_step(
        wave_parent,
        direction,
        step=config.swim_wave_step_rad_per_depth,
        lower=-math.pi,
        upper=math.pi,
    )
    wave_update = born & (trait_code == SWIM_WAVE)
    candidate.swim_wave_rad_per_depth = torch.where(
        wave_update, wave_child, candidate.swim_wave_rad_per_depth
    )

    parent_value = torch.where(
        joint_update,
        joint_parent,
        torch.where(frequency_update, frequency_parent, wave_parent),
    )
    child_value = torch.where(
        joint_update,
        joint_child,
        torch.where(frequency_update, frequency_child, wave_child),
    )
    mutated = born & ~unavailable
    ledger = MutationLedger(
        mutated=mutated,
        trait_code=trait_code,
        locus=torch.where(joint_update, locus, torch.full_like(locus, -1)),
        component=torch.full_like(locus, -1),
        parent_value=torch.where(mutated, parent_value, 0.0),
        child_value=torch.where(mutated, child_value, 0.0),
        unavailable=unavailable,
        mutation_count=mutated.to(torch.int64),
        event_applied=mutated[..., None],
        event_trait_code=trait_code[..., None],
        event_locus=torch.where(
            joint_update, locus, torch.full_like(locus, -1)
        )[..., None],
        event_component=torch.full_like(locus, -1)[..., None],
        event_parent_value=torch.where(mutated, parent_value, 0.0)[..., None],
        event_child_value=torch.where(mutated, child_value, 0.0)[..., None],
    )
    return MutationStep(candidate, ledger)
