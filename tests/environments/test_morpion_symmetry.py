"""State, action, relation and sampling contracts for D4 augmentation."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
from atomheart.games.morpion.canonical import apply_rooted_symmetry
from atomheart.games.morpion.dynamics import transform_action_rooted
from atomheart.games.morpion.state import Variant, initial_state, norm_seg

from chipiron.environments.morpion.action_geometry import morpion_action_points
from chipiron.environments.morpion.learning import (
    decode_morpion_state_ref_payload,
    iter_morpion_supervised_rows_from_path,
)
from chipiron.environments.morpion.players.evaluators.datasets.augmentation import (
    AugmentedMorpionDataset,
    training_symmetry,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MorpionEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.cached_index_schedule import (
    cached_index_schedule,
    shuffled_epoch_train_indices,
)
from chipiron.environments.morpion.symmetry import transform_morpion_state
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from tests.environments.test_morpion_entity_token_cache import _build_jsonl_rows_file

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=[Variant.TOUCHING_5T, Variant.DISJOINT_5D])
def state(request: pytest.FixtureRequest) -> MorpionState:
    """Construct an asymmetric legal board with endpoints and interior usage."""
    dynamics = MorpionDynamics()
    position = dynamics.wrap_atomheart_state(initial_state(request.param))
    rng = random.Random(31)
    for _ in range(12):
        position = dynamics.step(
            position, rng.choice(dynamics.all_legal_actions(position))
        ).next_state
    return position


@pytest.mark.parametrize("symmetry", range(8))
def test_symmetries_preserve_actions_and_transitions(
    state: MorpionState, symmetry: int
) -> None:
    """Raw legal action sets and resulting complete states commute with D4."""
    dynamics = MorpionDynamics()
    transformed = transform_morpion_state(state, symmetry)
    actions = dynamics.all_legal_actions(state)
    assert {transform_action_rooted(a, symmetry) for a in actions} == set(
        dynamics.all_legal_actions(transformed)
    )
    assert transformed.moves == state.moves
    assert transformed.variant == state.variant
    for action in actions[:3]:
        original_next = dynamics.step(state, action).next_state
        transformed_next = dynamics.step(
            transformed, transform_action_rooted(action, symmetry)
        ).next_state
        assert transformed_next == transform_morpion_state(original_next, symmetry)
    assert transform_morpion_state(
        replace(state, is_terminal=True), symmetry
    ).is_terminal


def test_group_closure_and_inverses(state: MorpionState) -> None:
    """The eight canonical maps form D4 on the complete state, not only points."""
    probes = ((0, 0), (1, 3), (-5, 2))
    signatures = {
        tuple(apply_rooted_symmetry(p, s) for p in probes): s for s in range(8)
    }
    assert len(signatures) == 8
    for a in range(8):
        inverse_found = False
        for b in range(8):
            signature = tuple(
                apply_rooted_symmetry(apply_rooted_symmetry(p, a), b) for p in probes
            )
            product = signatures[signature]
            assert transform_morpion_state(
                transform_morpion_state(state, a), b
            ) == transform_morpion_state(state, product)
            inverse_found |= product == 0
        assert inverse_found


@pytest.mark.parametrize("symmetry", range(8))
def test_relation_geometry_rebuilds_with_canonical_slot_order(
    state: MorpionState, symmetry: int
) -> None:
    """Rebuilding transforms relation endpoints and reverses slot IDs as needed."""
    transformed = transform_morpion_state(state, symmetry)
    converter = MorpionEntityTokenConverter()
    before = converter.state_to_layout(state)
    after = converter.state_to_layout(transformed)
    indices = {0: 0}
    for point, index in before.dot_index_by_point.items():
        indices[index] = after.dot_index_by_point[
            apply_rooted_symmetry(point, symmetry)
        ]
    for (a, b), index in before.edge_index_by_segment.items():
        segment = norm_seg(
            apply_rooted_symmetry(a, symmetry), apply_rooted_symmetry(b, symmetry)
        )
        indices[index] = after.edge_index_by_segment[segment]
    reversed_moves = set()
    for action, index in before.move_index_by_action.items():
        mapped = transform_action_rooted(action, symmetry)
        indices[index] = after.move_index_by_action[mapped]
        if (
            apply_rooted_symmetry(morpion_action_points(action)[0], symmetry)
            != morpion_action_points(mapped)[0]
        ):
            reversed_moves.add(index)
        assert morpion_action_points(mapped)[mapped[3]] == apply_rooted_symmetry(
            morpion_action_points(action)[action[3]], symmetry
        )
    relational = MorpionRelationalEntityTokenConverter()
    expected = set()
    for src, dst, kind in relational.state_to_tensors(state).relation_triples.tolist():
        if 1 <= kind <= 5 and src in reversed_moves:
            kind = 6 - kind
        elif 6 <= kind <= 10 and dst in reversed_moves:
            kind = 16 - kind
        expected.add((indices[src], indices[dst], kind))
    tensors = relational.state_to_tensors(transformed)
    assert expected == set(map(tuple, tensors.relation_triples.tolist()))
    assert torch.equal(
        tensors.token_tensor, relational.state_to_tensors(transformed).token_tensor
    )
    assert torch.equal(
        tensors.relation_triples,
        relational.state_to_tensors(transformed).relation_triples,
    )


def test_sampling_keeps_targets_split_validation_and_rng(tmp_path: Path) -> None:
    """Epoch requests vary training geometry while validation stays canonical."""
    path, _ = _build_jsonl_rows_file(
        tmp_path, target_values=tuple(float(x) for x in range(20))
    )
    rows = tuple(
        (
            MorpionState.from_atomheart_state(
                decode_morpion_state_ref_payload(r.state_ref_payload),
                is_terminal=r.is_terminal,
            ),
            r.target_value,
        )
        for r in iter_morpion_supervised_rows_from_path(path)
    )
    canonical = AugmentedMorpionDataset(rows, seed=2, augment=False)
    candidate = AugmentedMorpionDataset(rows, seed=2, augment=True)
    assert len(candidate) == len(canonical) == 20
    schedule = cached_index_schedule(row_count=20, validation_fraction=0.2)
    assert schedule.validation_indices == (4, 9, 14, 19)
    rng_state = random.getstate()
    choices = [
        training_symmetry(seed=2, epoch=e, row_index=i, enabled=True)
        for e in range(3)
        for i in schedule.train_indices
    ]
    assert choices == [
        training_symmetry(seed=2, epoch=e, row_index=i, enabled=True)
        for e in range(3)
        for i in schedule.train_indices
    ]
    assert set(choices) == set(range(8))
    assert random.getstate() == rng_state
    assert choices != [
        training_symmetry(seed=3, epoch=e, row_index=i, enabled=True)
        for e in range(3)
        for i in schedule.train_indices
    ]
    for index in range(len(rows)):
        assert torch.equal(
            candidate[index, 0].target_tensor, canonical[index, 0].target_tensor
        )
        a, b = candidate[index, -1], canonical[index, -1]
        assert torch.equal(a.input_tensor, b.input_tensor)
        assert torch.equal(a.auxiliary_input_tensors[0], b.auxiliary_input_tensors[0])
    order = shuffled_epoch_train_indices(
        train_indices=schedule.train_indices,
        shuffle=True,
        validation_seed=2,
        epoch_index=0,
    )
    assert set(order).isdisjoint(schedule.validation_indices)
