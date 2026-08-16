from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from sirrobin.genetics.develop import develop
from sirrobin.organisms.body_cache import commit_developed_births
from sirrobin.organisms.development import (
    calibrate_development_config,
    target_structure_cost_q,
)
from sirrobin.organisms.lifecycle import LifecycleRequest, settle_lifecycle
from sirrobin.organisms.mutation import (
    ATTACHMENT_ANGLE,
    ATTACHMENT_POSITION,
    SEGMENT_BUD,
    SEGMENT_RESHAPE,
    SEGMENT_VESTIGIAL,
    MutationConfig,
    commit_offspring_mutations,
    propose_offspring_mutations,
)
from sirrobin.runtime.reference_adapter import living_state_from_reference
from tools.run_world import _build_fixture_world

MORPHOLOGY_SWITCHES = {
    "joint_amplitude": False,
    "swim_frequency": False,
    "swim_wave": False,
    "segment_reshape": False,
    "attachment_position": False,
    "attachment_angle": False,
    "segment_bud": False,
    "segment_vestigial": False,
}


def _proposal(trait: str, *, seed: int = 17):
    world = _build_fixture_world(
        bodies=2,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    switches = dict(MORPHOLOGY_SWITCHES)
    switches[trait] = True
    config = MutationConfig(
        seed=seed,
        mutation_rate_per_locus=1.0,
        max_mutations_per_birth=1,
        **switches,
    )
    config.validate()
    requested = world.genotype.alive.clone()
    proposal = propose_offspring_mutations(
        world.genotype,
        world.body,
        requested,
        torch.tensor(9, dtype=torch.int64),
        config,
    )
    return world, config, proposal


def test_segment_reshape_is_one_small_log_proportional_change() -> None:
    world, config, proposal = _proposal("segment_reshape")
    changed = (
        proposal.genotype.node_log_axes_flu_m
        != world.genotype.node_log_axes_flu_m
    ).nonzero(as_tuple=False)

    assert proposal.ledger.trait_code.tolist() == [[SEGMENT_RESHAPE, 0]]
    assert changed.shape[0] == 1
    wi, ci, node, axis = changed[0].tolist()
    delta = (
        proposal.genotype.node_log_axes_flu_m[wi, ci, node, axis]
        - world.genotype.node_log_axes_flu_m[wi, ci, node, axis]
    )
    assert abs(float(delta)) == pytest.approx(
        config.segment_log_axis_step,
        abs=1.0e-7,
    )
    assert proposal.ledger.locus[wi, ci].item() == node
    assert proposal.ledger.component[wi, ci].item() == axis
    developed = develop(proposal.genotype)
    assert developed.truncated_candidate_count.tolist() == [[0, 0]]
    assert developed.mass_sim[0, 0].sum() != world.body.mass_sim[0, 0].sum()


def test_attachment_position_moves_one_coordinate_by_a_bounded_fraction() -> None:
    world, config, proposal = _proposal("attachment_position")
    changed = (
        proposal.genotype.edge_attach_parent_axes
        != world.genotype.edge_attach_parent_axes
    ).nonzero(as_tuple=False)

    assert proposal.ledger.trait_code.tolist() == [[ATTACHMENT_POSITION, 0]]
    assert changed.shape[0] == 1
    wi, ci, edge, axis = changed[0].tolist()
    delta = (
        proposal.genotype.edge_attach_parent_axes[wi, ci, edge, axis]
        - world.genotype.edge_attach_parent_axes[wi, ci, edge, axis]
    )
    assert abs(float(delta)) == pytest.approx(config.attachment_position_step)
    assert proposal.ledger.locus[wi, ci].item() == edge
    assert proposal.ledger.component[wi, ci].item() == axis


def test_attachment_angle_is_a_small_normalized_rotation() -> None:
    world, config, proposal = _proposal("attachment_angle")
    edge = int(proposal.ledger.locus[0, 0])
    before = world.genotype.edge_rot_flu[0, 0, edge]
    after = proposal.genotype.edge_rot_flu[0, 0, edge]

    assert proposal.ledger.trait_code.tolist() == [[ATTACHMENT_ANGLE, 0]]
    assert not torch.equal(after, before)
    assert torch.linalg.vector_norm(after).item() == pytest.approx(1.0)
    assert abs(float(proposal.ledger.child_value[0, 0])) == pytest.approx(
        config.attachment_angle_step_rad
    )
    assert torch.equal(
        proposal.genotype.edge_attach_parent_axes,
        world.genotype.edge_attach_parent_axes,
    )


def test_new_segment_is_a_connected_small_unported_bud() -> None:
    world, config, proposal = _proposal("segment_bud")
    parent = world.genotype
    child = proposal.genotype
    new_nodes = child.node_mask & ~parent.node_mask
    new_edges = child.edge_mask & ~parent.edge_mask

    assert proposal.ledger.trait_code.tolist() == [[SEGMENT_BUD, 0]]
    assert new_nodes.sum().item() == 1
    assert new_edges.sum().item() == 1
    node = int(new_nodes[0, 0].nonzero()[0])
    edge = int(new_edges[0, 0].nonzero()[0])
    source = int(child.edge_src[0, 0, edge])
    assert int(child.edge_dst[0, 0, edge]) == node
    assert torch.all(
        child.node_log_axes_flu_m[0, 0, node].exp()
        <= parent.node_log_axes_flu_m[0, 0, source].exp()
        * config.bud_axis_fraction
        * (1.0 + 1.0e-6)
    )
    assert not bool(child.node_intake[0, 0, node])
    assert not bool(child.node_sense[0, 0, node])
    assert child.node_joint_amp_rad[0, 0, node].item() == 0.0
    developed = develop(child)
    assert developed.truncated_candidate_count.tolist() == [[0, 0]]
    assert developed.seg_mask[0, 0].sum() == world.body.seg_mask[0, 0].sum() + 1


def test_vestigial_mutation_shrinks_before_it_can_remove_a_leaf() -> None:
    world, config, proposal = _proposal("segment_vestigial")
    node = int(proposal.ledger.locus[0, 0])

    assert proposal.ledger.trait_code.tolist() == [[SEGMENT_VESTIGIAL, 0]]
    assert torch.equal(proposal.genotype.node_mask, world.genotype.node_mask)
    assert torch.allclose(
        proposal.genotype.node_log_axes_flu_m[0, 0, node],
        world.genotype.node_log_axes_flu_m[0, 0, node]
        + math.log(config.vestigial_axis_fraction),
    )

    tiny = world.genotype
    tiny.node_log_axes_flu_m = tiny.node_log_axes_flu_m.clone()
    tiny.node_log_axes_flu_m[0, 0, node] = (
        tiny.node_log_axes_flu_m[0, 0, 0]
        + math.log(config.vestigial_remove_axis_fraction * 0.5)
    )
    removed = propose_offspring_mutations(
        tiny,
        develop(tiny),
        tiny.alive,
        torch.tensor(9, dtype=torch.int64),
        config,
    )

    assert not bool(removed.genotype.node_mask[0, 0, node])
    assert removed.genotype.node_mask[0, 0].sum() == tiny.node_mask[0, 0].sum() - 1
    assert removed.genotype.edge_mask[0, 0].sum() == tiny.edge_mask[0, 0].sum() - 1
    assert develop(removed.genotype).truncated_candidate_count.tolist() == [[0, 0]]


def test_morphology_proposal_is_one_full_compiled_graph() -> None:
    world, config, expected = _proposal("segment_bud")
    compiled = torch.compile(
        propose_offspring_mutations,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )

    actual = compiled(
        world.genotype,
        world.body,
        world.genotype.alive,
        torch.tensor(9, dtype=torch.int64),
        config,
    )

    assert torch.equal(actual.genotype.node_mask, expected.genotype.node_mask)
    assert torch.equal(actual.genotype.edge_mask, expected.genotype.edge_mask)
    assert torch.equal(actual.ledger.trait_code, expected.ledger.trait_code)


def test_births_can_receive_zero_or_multiple_mutation_events() -> None:
    world = _build_fixture_world(
        bodies=2,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    switches = dict(MORPHOLOGY_SWITCHES)
    switches["segment_reshape"] = True
    zero = propose_offspring_mutations(
        world.genotype,
        world.body,
        world.genotype.alive,
        torch.tensor(9, dtype=torch.int64),
        MutationConfig(
            seed=17,
            mutation_rate_per_locus=0.0,
            max_mutations_per_birth=3,
            **switches,
        ),
    )
    multiple = propose_offspring_mutations(
        world.genotype,
        world.body,
        world.genotype.alive,
        torch.tensor(9, dtype=torch.int64),
        MutationConfig(
            seed=17,
            mutation_rate_per_locus=1.0,
            max_mutations_per_birth=3,
            **switches,
        ),
    )

    assert zero.ledger.mutation_count.tolist() == [[0, 0]]
    assert torch.equal(
        zero.genotype.node_log_axes_flu_m,
        world.genotype.node_log_axes_flu_m,
    )
    assert multiple.ledger.mutation_count.tolist() == [[3, 0]]
    assert multiple.ledger.event_trait_code[0, 0].tolist() == [
        SEGMENT_RESHAPE,
        SEGMENT_RESHAPE,
        SEGMENT_RESHAPE,
    ]


def test_candidate_is_developed_priced_and_only_then_committed() -> None:
    world, _, proposal = _proposal("segment_bud")
    state = living_state_from_reference(world)
    candidate_body = develop(proposal.genotype)
    development_config = calibrate_development_config(
        state.population,
        state.body,
    )
    child_structure_q = target_structure_cost_q(
        candidate_body,
        development_config,
    )
    lifecycle = settle_lifecycle(
        state.population,
        LifecycleRequest(
            death=torch.zeros_like(state.population.alive),
            birth=state.population.alive,
            child_structure_q=child_structure_q,
            child_reserve_q=torch.full_like(child_structure_q, 100),
            birth_release_energy_q=torch.zeros_like(child_structure_q),
            time_s=torch.tensor([1.0], dtype=torch.float64),
        ),
    )
    committed = commit_offspring_mutations(
        state.genotype,
        proposal,
        lifecycle.state,
        lifecycle.ledger,
    )
    committed_body = commit_developed_births(
        state.body,
        candidate_body,
        committed.genotype,
        lifecycle.state,
        lifecycle.ledger,
    )
    child_slot = int(lifecycle.ledger.born[0].nonzero()[0])
    parent_slot = int(lifecycle.ledger.parent_slot_for_child[0, child_slot])

    assert lifecycle.state.structure_q[0, child_slot] == child_structure_q[0, parent_slot]
    assert torch.equal(
        committed.genotype.node_mask[0, child_slot],
        proposal.genotype.node_mask[0, parent_slot],
    )
    assert torch.equal(
        committed_body.semi_axes_flu_m[0, child_slot],
        candidate_body.semi_axes_flu_m[0, parent_slot],
    )
    assert committed.ledger.trait_code[0, child_slot] == SEGMENT_BUD
    assert committed_body.seg_mask[0, child_slot].sum() == state.body.seg_mask[0, 0].sum() + 1


def test_unaffordable_candidate_is_not_replaced_by_a_clone() -> None:
    world, _, proposal = _proposal("segment_bud")
    state = living_state_from_reference(world)
    candidate_body = develop(proposal.genotype)
    development_config = calibrate_development_config(
        state.population,
        state.body,
    )
    child_structure_q = target_structure_cost_q(candidate_body, development_config)
    population = replace(
        state.population,
        reserve_q=torch.where(
            state.population.alive,
            child_structure_q,
            state.population.reserve_q,
        ),
    )
    lifecycle = settle_lifecycle(
        population,
        LifecycleRequest(
            death=torch.zeros_like(population.alive),
            birth=population.alive,
            child_structure_q=child_structure_q,
            child_reserve_q=torch.full_like(child_structure_q, 100),
            birth_release_energy_q=torch.zeros_like(child_structure_q),
            time_s=torch.tensor([1.0], dtype=torch.float64),
        ),
    )
    committed = commit_offspring_mutations(
        state.genotype,
        proposal,
        lifecycle.state,
        lifecycle.ledger,
    )

    assert lifecycle.ledger.accepted_births.tolist() == [0]
    assert lifecycle.ledger.unfunded_rejections.tolist() == [1]
    assert torch.equal(committed.genotype.alive, state.genotype.alive)
    assert torch.equal(committed.genotype.node_mask, state.genotype.node_mask)
