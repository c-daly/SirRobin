from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sirrobin.observe.runtime_snapshot import stage_runtime_events
from sirrobin.organisms.behavior import BehaviorConfig, request_living_intent
from sirrobin.organisms.development import (
    calibrate_development_config,
    initialize_development_state,
    validate_development_state,
)
from sirrobin.organisms.feeding import FeedingConfig
from sirrobin.organisms.metabolism import MetabolismConfig
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.organisms.mutation import MutationConfig
from sirrobin.organisms.state import PopulationState
from sirrobin.physics.phase_response import PhaseWindowConfig
from sirrobin.runtime.config import (
    LivingRuntimeConfig,
    validate_living_runtime_config,
)
from sirrobin.runtime.material import total_matter_q
from sirrobin.runtime.motion_state import developed_support_radius_m
from sirrobin.runtime.session import RuntimeSession
from sirrobin.runtime.state import LivingState, validate_living_state
from sirrobin.runtime.step import (
    LivingIntervalInputs,
    advance_living_interval,
)
from tools.run_world import (
    LIVING_MATERIAL_ENERGY_CONFIG,
    _build_fixture_world,
)
from tools.runtime_unity import runtime_events


def _fixture(
    device: torch.device | None = None,
) -> tuple[LivingState, LivingIntervalInputs, LivingRuntimeConfig]:
    if device is None:
        device = torch.device("cpu")
    world = _build_fixture_world(
        bodies=3,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=device,
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    alive = world.genotype.alive
    zeros_i64 = torch.zeros_like(alive, dtype=torch.int64)
    zeros_f64 = torch.zeros_like(alive, dtype=torch.float64)
    population = PopulationState(
        alive=alive,
        stable_id=world.genotype.stable_id,
        parent_id=zeros_i64,
        generation=zeros_i64,
        born_at_s=zeros_f64,
        structure_q=world.creature_material.structure_q,
        reserve_q=world.creature_material.reserve_q,
        intake_carry_mol=world.creature_material.intake_carry_mol,
        assimilation_carry_q=world.creature_material.assimilation_carry_q,
        maintenance_carry_j=world.creature_material.maintenance_carry_j,
        next_stable_id=world.next_stable_id,
    )
    state = LivingState(
        population,
        world.genotype,
        world.body,
        initialize_development_state(population, world.body),
        world.live_state,
        world.economy_state,
        total_matter_q(world.economy_state, population),
    )
    config = LivingRuntimeConfig(
        economy=world.economy_config,
        live=world.live_config,
        motion=PhaseWindowConfig(0.1, stages=4, phase_samples=2),
        behavior=BehaviorConfig(1.0),
        feeding=FeedingConfig(
            interval_s=0.1,
            q_mass_mol=world.economy_config.q_mass_mol,
            capture_efficiency=0.5,
            assimilation_efficiency=0.5,
            producer_j_per_q=LIVING_MATERIAL_ENERGY_CONFIG.producer_j_per_q,
            reserve_j_per_q=LIVING_MATERIAL_ENERGY_CONFIG.reserve_j_per_q,
        ),
        metabolism=MetabolismConfig(
            interval_s=0.1,
            maintenance_w_per_kg=0.01,
            chemical_to_mechanical_efficiency=1.0,
            reserve_j_per_q=LIVING_MATERIAL_ENERGY_CONFIG.reserve_j_per_q,
        ),
        mortality=MortalityConfig(100.0, 100.0, seed=7),
        mutation=MutationConfig(
            seed=11,
            mutation_rate_per_locus=1.0,
            max_mutations_per_birth=1,
        ),
        development=calibrate_development_config(population, world.body),
        child_initial_reserve_q=100,
    )
    inputs = LivingIntervalInputs(
        fluid=world.fluid,
        requested_effort=torch.where(
            alive,
            torch.ones_like(alive, dtype=torch.float32),
            torch.zeros_like(alive, dtype=torch.float32),
        ),
        birth_requested=alive.clone(),
    )
    validate_living_runtime_config(config)
    validate_living_state(state, config.economy)
    return state, inputs, config


def test_session_replays_dense_candidates_on_a_birth_request() -> None:
    state, inputs, config = _fixture()
    expected = state
    for _ in range(2):
        expected = advance_living_interval(expected, inputs, config).state
    session = RuntimeSession(
        state,
        config,
        compile_motion=False,
        compile_domains=True,
        compile_backend="eager",
    )

    actual = session.advance_chunk(inputs, intervals=2)

    assert torch.equal(actual.state.population.alive, expected.population.alive)
    assert torch.equal(
        actual.state.population.reserve_q,
        expected.population.reserve_q,
    )
    assert torch.equal(actual.state.genotype.node_mask, expected.genotype.node_mask)
    assert actual.summary is not None
    assert actual.summary.births.tolist() == [2]
    assert actual.last_interval.candidate_work_deferred.tolist() == [False]
    assert actual.last_interval.candidate_slots_evaluated.tolist() == [3]
    assert actual.last_interval.candidate_replay_required.tolist() == [False]
    assert actual.last_interval.matter.books_closed.tolist() == [True]


def test_session_defers_zero_request_candidate_work_without_replay() -> None:
    state, inputs, config = _fixture()
    inputs = replace(inputs, birth_requested=torch.zeros_like(inputs.birth_requested))
    expected = advance_living_interval(state, inputs, config)
    session = RuntimeSession(
        state,
        config,
        compile_motion=False,
        compile_domains=True,
        compile_backend="eager",
    )

    actual = session.advance_chunk(inputs, intervals=1)

    assert torch.equal(
        actual.state.population.reserve_q,
        expected.state.population.reserve_q,
    )
    assert torch.equal(actual.state.genotype.node_mask, expected.state.genotype.node_mask)
    assert actual.last_interval.candidate_work_deferred.tolist() == [True]
    assert actual.last_interval.candidate_slots_evaluated.tolist() == [0]
    assert actual.last_interval.candidate_replay_required.tolist() == [False]
    assert actual.last_interval.matter.books_closed.tolist() == [True]


def test_feeding_retry_uses_dense_candidates_for_a_new_late_request() -> None:
    state, inputs, config = _fixture()
    economy = state.economy.clone()
    transferred_q = state.population.reserve_q.sum(dtype=torch.int64)
    economy.nd_q[0, 0, 0, 0] += transferred_q
    population = replace(
        state.population,
        reserve_q=torch.zeros_like(state.population.reserve_q),
    )
    state = replace(state, population=population, economy=economy)
    config = replace(
        config,
        metabolism=replace(config.metabolism, maintenance_w_per_kg=6.0),
    )
    validate_living_state(state, config.economy)
    session = RuntimeSession(
        state,
        config,
        compile_motion=False,
        compile_domains=True,
        optimistic_motion=False,
        compile_backend="eager",
    )
    feeding = session._kernels.feeding
    fast_candidates = session._kernels.candidates
    dense_candidates = session._robust_kernels.candidates
    candidate_modes: list[str] = []

    def controlled_feeding(*args):
        population_before = args[0]
        producer_before = args[1]
        dissolved_before = args[2]
        velocity = torch.ones_like(args[4])
        feeding_config = args[-1]
        step = feeding(*args[:4], velocity, *args[5:])
        if feeding_config.allocation_rounds > 1:
            return step
        slot_zeros = torch.zeros_like(step.ledger.actual_debit_q)
        cell_zeros = torch.zeros_like(step.ledger.producer_debit_by_cell_q)
        ledger = replace(
            step.ledger,
            actual_debit_q=slot_zeros,
            reserve_credit_q=slot_zeros,
            dissolved_return_q=slot_zeros,
            producer_debit_by_cell_q=cell_zeros,
            dissolved_credit_by_cell_q=cell_zeros,
            producer_chemical_input_j=torch.zeros_like(
                step.ledger.producer_chemical_input_j
            ),
            reserve_chemical_credit_j=torch.zeros_like(
                step.ledger.reserve_chemical_credit_j
            ),
            assimilation_heat_j=torch.zeros_like(
                step.ledger.assimilation_heat_j
            ),
            allocation_rounds_exhausted=population_before.alive,
            transaction_committed=torch.zeros_like(
                step.ledger.transaction_committed
            ),
            invalid=torch.ones_like(step.ledger.invalid),
        )
        return replace(
            step,
            population=population_before,
            producer_q=producer_before,
            dissolved_q=dissolved_before,
            ledger=ledger,
        )

    def counted_candidates(label, function):
        def invoke(*args, **kwargs):
            candidate_modes.append(label)
            return function(*args, **kwargs)

        return invoke

    session._kernels = replace(
        session._kernels,
        feeding=controlled_feeding,
        candidates=counted_candidates("deferred", fast_candidates),
    )
    session._robust_kernels = replace(
        session._robust_kernels,
        feeding=controlled_feeding,
        candidates=counted_candidates("dense", dense_candidates),
    )

    def late_request_provider(candidate):
        birth_requested = candidate.population.alive & (
            candidate.economy.step > 0
        )
        return candidate, replace(inputs, birth_requested=birth_requested), None

    actual = session._advance_with_provider(late_request_provider, intervals=2)

    assert candidate_modes == ["deferred", "deferred", "dense", "dense"]
    assert actual.invalid.tolist() == [False]
    assert actual.last_interval.candidate_work_deferred.tolist() == [False]
    assert actual.last_interval.candidate_replay_required.tolist() == [False]
    assert actual.last_interval.matter.books_closed.tolist() == [True]


def test_complete_interval_closes_matter_and_commits_a_mutated_paid_birth() -> None:
    state, inputs, config = _fixture()

    step = advance_living_interval(state, inputs, config)

    assert step.ledger.matter.books_closed.tolist() == [True]
    assert step.ledger.matter.before_q.tolist() == step.ledger.matter.after_q.tolist()
    assert step.ledger.invalid.tolist() == [False]
    assert step.ledger.organisms.lifecycle.ledger.accepted_births.tolist() == [1]
    assert step.ledger.organisms.lifecycle.ledger.born.sum().item() == 1
    assert step.ledger.mutation.ledger.mutated.sum().item() == 1
    assert step.ledger.candidate_work_deferred.tolist() == [False]
    assert step.ledger.candidate_slots_evaluated.tolist() == [3]
    assert step.ledger.candidate_replay_required.tolist() == [False]
    assert step.state.population.alive.sum().item() == 2
    assert torch.equal(step.state.body.alive, step.state.population.alive)
    assert torch.equal(step.state.body.stable_id, step.state.population.stable_id)
    validate_development_state(
        step.state.development,
        step.state.population,
        step.state.body,
    )
    born = step.ledger.organisms.lifecycle.ledger.born
    accepted_parent = step.ledger.organisms.lifecycle.ledger.accepted_parent
    child_slot = int(born[0].nonzero()[0])
    parent_slot = int(accepted_parent[0].nonzero()[0])
    assert torch.count_nonzero(step.state.motion.gait_time_s[born]) == 0
    assert torch.count_nonzero(step.state.motion.velocity_rel_water_enu_m_s[born]) > 0
    assert not torch.equal(
        step.state.motion.position_enu_m[born],
        step.state.motion.position_enu_m[accepted_parent],
    )
    periods = step.state.motion.position_enu_m.new_tensor(
        [config.geometry.lx_m, config.geometry.ly_m]
    )
    displacement_xy = (
        step.state.motion.position_enu_m[0, child_slot, :2]
        - step.state.motion.position_enu_m[0, parent_slot, :2]
    )
    minimum_image_xy = torch.remainder(
        displacement_xy + 0.5 * periods,
        periods,
    ) - 0.5 * periods
    support_radius = developed_support_radius_m(step.state.body)
    required_separation = (
        support_radius[0, parent_slot]
        + support_radius[0, child_slot]
        + config.birth_separation_clearance_m
    )
    assert torch.linalg.vector_norm(minimum_image_xy) >= required_separation
    assert torch.allclose(
        step.ledger.release.parent_impulse_enu_ns.sum(dim=1)
        + step.ledger.release.child_impulse_enu_ns.sum(dim=1),
        torch.zeros((1, 3)),
    )
    release_q = (
        step.ledger.organisms.lifecycle.ledger.birth_release_energy_return_q
    )
    assert release_q.sum() > 0
    assert step.ledger.energy.birth_release_chemical_input_j.sum().item() == (
        release_q.sum().item() * config.metabolism.reserve_j_per_q
    )
    assert torch.allclose(
        step.ledger.energy.birth_release_heat_j,
        step.ledger.energy.birth_release_chemical_input_j
        - step.ledger.energy.birth_release_kinetic_delta_j,
    )
    assert bool((step.ledger.energy.birth_release_heat_j >= 0.0).all())
    assert step.ledger.energy.birth_construction_heat_j.sum().item() == (
        step.ledger.organisms.lifecycle.ledger.birth_structure_transfer_q.sum().item()
        * config.metabolism.reserve_j_per_q
    )
    assert not bool(step.state.motion.heading_initialized[born].any())
    assert not bool(step.state.motion.desired_heading_enu[born].any())
    first_child_intent = request_living_intent(
        step.state.population,
        step.state.body,
        step.state.motion,
        step.state.economy.bp_q,
        config.geometry,
        config.live,
        config.behavior,
        q_mass_mol=config.economy.q_mass_mol,
    )
    assert first_child_intent.sampled_producer_mol_m3[born] > 0.0
    assert bool(first_child_intent.locomoting[born].all())
    validate_living_state(step.state, config.economy)


def test_impossible_birth_release_blocks_only_the_birth() -> None:
    state, inputs, config = _fixture()
    config = replace(
        config,
        birth_separation_clearance_m=min(
            config.geometry.lx_m,
            config.geometry.ly_m,
        ),
    )

    session = RuntimeSession(
        state,
        config,
        compile_motion=False,
        compile_domains=False,
        optimistic_motion=False,
        optimistic_feeding=False,
        optimistic_candidates=False,
    )

    chunk = session.advance_chunk(inputs, intervals=1)

    lifecycle = chunk.last_interval.organisms.lifecycle.ledger
    assert lifecycle.accepted_births.tolist() == [0]
    assert not bool(lifecycle.born.any())
    assert chunk.state.population.alive.sum().item() == 1
    assert chunk.state.economy.step.item() == state.economy.step.item() + 1
    assert chunk.last_interval.matter.books_closed.tolist() == [True]
    assert chunk.invalid.tolist() == [False]
    validate_living_state(chunk.state, config.economy)


def test_morphology_birth_is_developed_and_paid_before_commit() -> None:
    state, inputs, config = _fixture()
    config = replace(
        config,
        mutation=MutationConfig(
            seed=29,
            joint_amplitude=False,
            swim_frequency=False,
            swim_wave=False,
            segment_reshape=False,
            attachment_position=False,
            attachment_angle=False,
            segment_bud=True,
            segment_vestigial=False,
            mutation_rate_per_locus=1.0,
            max_mutations_per_birth=1,
        ),
    )
    parent_segments = int(state.body.seg_mask[0, 0].sum())

    step = advance_living_interval(state, inputs, config)

    lifecycle = step.ledger.organisms.lifecycle.ledger
    child_slot = int(lifecycle.born[0].nonzero()[0])
    parent_slot = int(lifecycle.parent_slot_for_child[0, child_slot])
    assert step.ledger.mutation.ledger.trait_code[0, child_slot] == 7
    assert int(step.state.body.seg_mask[0, child_slot].sum()) == parent_segments + 1
    assert lifecycle.birth_structure_transfer_q[0, parent_slot] == (
        step.state.population.structure_q[0, child_slot]
    )
    validate_development_state(
        step.state.development,
        step.state.population,
        step.state.body,
    )
    assert step.ledger.matter.books_closed.tolist() == [True]
    event_snapshot = stage_runtime_events(step.state, step.ledger)
    assert event_snapshot.mutation_count[0, child_slot] == 1
    assert "segment_bud" in runtime_events(event_snapshot)[0]


def test_paid_birth_without_a_mutation_is_observed_as_such() -> None:
    state, inputs, config = _fixture()
    config = replace(
        config,
        mutation=replace(config.mutation, mutation_rate_per_locus=0.0),
    )

    step = advance_living_interval(state, inputs, config)
    event_snapshot = stage_runtime_events(step.state, step.ledger)
    child_slot = int(step.ledger.organisms.lifecycle.ledger.born[0].nonzero()[0])

    assert event_snapshot.mutation_count[0, child_slot] == 0
    assert runtime_events(event_snapshot) == [
        f"creature 1 reproduced: child "
        f"{int(step.state.population.stable_id[0, child_slot])}; no mutation"
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_cuda_interval_commits_paid_birth_with_exact_books() -> None:
    state, inputs, config = _fixture(torch.device("cuda"))
    session = RuntimeSession(
        state,
        config,
        compile_motion=True,
        compile_domains=True,
    )

    chunk = session.advance_chunk(inputs, intervals=1)
    lifecycle = chunk.last_interval.organisms.lifecycle.ledger

    assert lifecycle.accepted_births.tolist() == [1]
    assert lifecycle.born.sum().item() == 1
    assert chunk.last_interval.mutation.ledger.mutated.sum().item() == 1
    assert chunk.last_interval.candidate_work_deferred.tolist() == [False]
    assert chunk.last_interval.candidate_slots_evaluated.tolist() == [3]
    assert chunk.last_interval.candidate_replay_required.tolist() == [False]
    assert chunk.last_interval.matter.books_closed.tolist() == [True]
    assert chunk.last_interval.invalid.tolist() == [False]
    assert torch.equal(chunk.state.body.alive, chunk.state.population.alive)
    assert torch.equal(chunk.state.body.stable_id, chunk.state.population.stable_id)
    born = lifecycle.born
    assert torch.count_nonzero(chunk.state.motion.gait_time_s[born]) == 0
    assert torch.count_nonzero(chunk.state.motion.velocity_rel_water_enu_m_s[born]) > 0
    validate_living_state(chunk.state, config.economy)


def test_old_age_death_returns_all_tracked_material_and_clears_motion() -> None:
    state, inputs, config = _fixture()
    inputs = replace(inputs, birth_requested=torch.zeros_like(inputs.birth_requested))
    config = replace(config, mortality=MortalityConfig(0.05, 0.05, seed=7))

    step = advance_living_interval(state, inputs, config)

    lifecycle = step.ledger.organisms.lifecycle.ledger
    assert lifecycle.died.sum().item() == 1
    assert lifecycle.death_structure_return_q.sum().item() > 0
    assert lifecycle.death_reserve_return_q.sum().item() > 0
    assert step.state.population.alive.sum().item() == 0
    assert torch.count_nonzero(step.state.motion.velocity_rel_water_enu_m_s) == 0
    assert torch.count_nonzero(step.state.motion.yaw_momentum_kg_m2_s) == 0
    assert step.ledger.returns.credit_by_cell_q.sum().item() == (
        step.ledger.organisms.metabolism.ledger.maintenance_return_q.sum()
        + lifecycle.death_structure_return_q.sum()
        + lifecycle.death_reserve_return_q.sum()
    ).item()
    assert step.ledger.matter.books_closed.tolist() == [True]
    assert step.ledger.invalid.tolist() == [False]
    validate_living_state(step.state, config.economy)


def test_pre_step_material_mint_cannot_be_redefined_as_a_new_baseline() -> None:
    state, inputs, config = _fixture()
    minted_reserve = state.population.reserve_q.clone()
    minted_reserve[0, 0] += 1
    tampered = replace(
        state,
        population=replace(state.population, reserve_q=minted_reserve),
    )

    step = advance_living_interval(tampered, inputs, config)

    assert step.ledger.matter.before_q.tolist() == [
        int(state.expected_matter_q.item()) + 1
    ]
    assert step.ledger.matter.books_closed.tolist() == [False]
    assert step.ledger.invalid.tolist() == [True]


def test_session_stage_composition_matches_eager_and_accepts_once_per_chunk() -> None:
    state, inputs, config = _fixture()
    inputs = replace(inputs, birth_requested=torch.zeros_like(inputs.birth_requested))
    expected = advance_living_interval(state, inputs, config)
    session = RuntimeSession(
        state,
        config,
        compile_motion=True,
        compile_backend="eager",
    )

    actual = session.advance_chunk(inputs, intervals=1)

    assert torch.equal(actual.state.population.reserve_q, expected.state.population.reserve_q)
    assert torch.equal(actual.state.economy.nd_q, expected.state.economy.nd_q)
    assert torch.allclose(
        actual.state.motion.position_enu_m,
        expected.state.motion.position_enu_m,
    )
    assert actual.last_interval.matter.books_closed.tolist() == [True]
    assert session.state is actual.state


def test_autonomous_session_composes_behavior_and_paid_birth() -> None:
    state, fixture_inputs, config = _fixture()
    behavior = request_living_intent(
        state.population,
        state.body,
        state.motion,
        state.economy.bp_q,
        config.geometry,
        config.live,
        config.behavior,
        q_mass_mol=config.economy.q_mass_mol,
    )
    expected = advance_living_interval(
        replace(state, motion=behavior.motion),
        LivingIntervalInputs(
            fluid=fixture_inputs.fluid,
            requested_effort=behavior.requested_effort_fraction,
            birth_requested=behavior.birth_requested,
        ),
        config,
    )
    session = RuntimeSession(
        state,
        config,
        compile_motion=True,
        compile_domains=True,
        compile_backend="eager",
    )

    actual = session.advance_autonomous_chunk(fixture_inputs.fluid, intervals=1)

    assert actual.last_behavior is not None
    assert actual.last_behavior.birth_requested.tolist() == [[True, False, False]]
    assert torch.equal(actual.state.population.alive, expected.state.population.alive)
    assert torch.equal(
        actual.state.population.reserve_q,
        expected.state.population.reserve_q,
    )
    assert actual.last_interval.matter.books_closed.tolist() == [True]
    assert actual.last_interval.candidate_work_deferred.tolist() == [False]
    assert actual.last_interval.candidate_slots_evaluated.tolist() == [3]
    assert actual.last_interval.candidate_replay_required.tolist() == [False]


@pytest.mark.parametrize(
    "optimistic_candidates",
    [0, 1, 1.0, "yes", None],
)
def test_session_rejects_nonboolean_optimistic_candidates(
    optimistic_candidates: object,
) -> None:
    state, _, config = _fixture()

    with pytest.raises(TypeError, match="optimistic candidates"):
        RuntimeSession(
            state,
            config,
            compile_motion=False,
            optimistic_candidates=optimistic_candidates,  # type: ignore[arg-type]
        )


def test_autonomous_prewarm_does_not_advance_authoritative_state() -> None:
    state, fixture_inputs, config = _fixture()
    initial_position = state.motion.position_enu_m.clone()
    initial_reserve = state.population.reserve_q.clone()
    initial_fields = tuple(value.clone() for value in state.economy.reservoirs)
    session = RuntimeSession(
        state,
        config,
        compile_motion=True,
        compile_domains=True,
        compile_backend="eager",
    )

    session.prewarm_autonomous(fixture_inputs.fluid)

    assert session.state is state
    assert state.economy.step.item() == 0
    assert torch.equal(state.motion.position_enu_m, initial_position)
    assert torch.equal(state.population.reserve_q, initial_reserve)
    assert all(
        torch.equal(current, initial)
        for current, initial in zip(
            state.economy.reservoirs,
            initial_fields,
            strict=True,
        )
    )


def test_autonomous_prewarm_exercises_fast_and_robust_specializations() -> None:
    state, fixture_inputs, config = _fixture()
    session = RuntimeSession(
        state,
        config,
        compile_motion=False,
        compile_domains=False,
    )
    motion_paths: list[str] = []
    feeding_rounds: list[int] = []
    fast_motion = session._kernels.motion
    robust_motion = session._robust_kernels.motion
    feeding = session._kernels.feeding

    def counted_motion(label, function):
        def invoke(*args, **kwargs):
            motion_paths.append(label)
            return function(*args, **kwargs)

        return invoke

    def counted_feeding(*args, **kwargs):
        config_argument = args[-1]
        feeding_rounds.append(config_argument.allocation_rounds)
        return feeding(*args, **kwargs)

    session._kernels = replace(
        session._kernels,
        motion=counted_motion("fast", fast_motion),
        feeding=counted_feeding,
    )
    session._robust_kernels = replace(
        session._robust_kernels,
        motion=counted_motion("robust", robust_motion),
        feeding=counted_feeding,
    )

    session.prewarm_autonomous(fixture_inputs.fluid)

    assert motion_paths == ["fast", "robust"]
    assert feeding_rounds == [1, config.feeding.allocation_rounds]
    assert session.state is state


def test_session_does_not_publish_an_invalid_candidate_chunk() -> None:
    state, inputs, config = _fixture()
    session = RuntimeSession(state, config, compile_motion=False)
    malformed = replace(
        inputs,
        requested_effort=torch.full_like(inputs.requested_effort, float("nan")),
    )

    try:
        session.advance_chunk(malformed, intervals=1)
    except ValueError as error:
        assert "requested effort" in str(error)
    else:
        raise AssertionError("an invalid candidate chunk was accepted")

    assert session.state is state


@pytest.mark.parametrize("optimistic_motion", [True, False])
def test_session_resolves_unfunded_request_through_exact_effort_options(
    optimistic_motion: bool,
) -> None:
    state, inputs, config = _fixture()
    economy = state.economy.clone()
    transferred_q = state.population.reserve_q.sum(dtype=torch.int64)
    economy.nd_q[0, 0, 0, 0] += transferred_q
    population = replace(
        state.population,
        reserve_q=torch.zeros_like(state.population.reserve_q),
    )
    state = replace(state, population=population, economy=economy)
    inputs = replace(inputs, birth_requested=torch.zeros_like(inputs.birth_requested))
    config = replace(
        config,
        metabolism=replace(config.metabolism, maintenance_w_per_kg=0.0),
    )
    validate_living_state(state, config.economy)
    expected = advance_living_interval(state, inputs, config)
    session = RuntimeSession(
        state,
        config,
        compile_motion=False,
        optimistic_motion=optimistic_motion,
    )

    actual = session.advance_chunk(inputs, intervals=1)

    assert expected.ledger.motion.ledger.selected.effort_fraction.tolist() == [
        [0.0, 0.0, 0.0]
    ]
    assert torch.equal(
        actual.state.motion.position_enu_m,
        expected.state.motion.position_enu_m,
    )
    assert torch.equal(
        actual.state.population.reserve_q,
        expected.state.population.reserve_q,
    )
    assert actual.last_interval.motion_funding_unresolved.tolist() == [False]
    assert actual.last_interval.matter.books_closed.tolist() == [True]
    assert session.optimistic_motion_enabled is optimistic_motion
