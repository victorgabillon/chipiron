"""Tests for sparse structural relations over Morpion entity tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

from chipiron.environments.morpion.learning import (
    decode_morpion_state_ref_payload,
    load_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionRelationalEntityTokenSupervisedDataset,
    MorpionRelationalEntityTokenSupervisedDatasetArgs,
    MorpionRelationalEntityTokenSupervisedSample,
    collate_morpion_relational_entity_token_supervised_samples,
    process_morpion_supervised_row_to_relational_entity_token_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    MORPION_ENTITY_TOKEN_FEATURE_NAMES,
    MorpionEntityRelationType,
    MorpionEntityTokenConverter,
    MorpionRelationalEntityTokenConverter,
    canonical_segment,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    DIRECTIONS,
)
from chipiron.environments.morpion.types import MorpionDynamics
from tests.environments.test_morpion_entity_tokens import (
    _build_rows_file,
    _make_one_step_state,
    _make_standard_state,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.types import MorpionAction, MorpionState


class _FixedActionDynamics:
    """Expose one structural action, including topologies forbidden in play."""

    def __init__(self, action: MorpionAction) -> None:
        self.action = action

    def all_legal_actions(self, _state: MorpionState) -> tuple[MorpionAction, ...]:
        """Return the fixed action used to exercise relation geometry."""
        return (self.action,)


def test_relation_enum_contract_is_exact() -> None:
    """Relation type zero and all fifteen active IDs should remain stable."""
    assert MORPION_ENTITY_RELATION_SCHEMA == "morpion_entity_relations_v1"
    assert MORPION_ENTITY_RELATION_TYPE_COUNT == 16
    assert [relation.value for relation in MorpionEntityRelationType] == list(range(16))
    assert MorpionEntityRelationType.NO_RELATION == 0
    assert MorpionEntityRelationType.MOVE_TO_DOT_SLOT_0 == 1
    assert MorpionEntityRelationType.MOVE_TO_DOT_SLOT_4 == 5
    assert MorpionEntityRelationType.DOT_SLOT_0_OF_MOVE == 6
    assert MorpionEntityRelationType.DOT_SLOT_4_OF_MOVE == 10
    assert MorpionEntityRelationType.EDGE_TOUCHES_DOT == 11
    assert MorpionEntityRelationType.DOT_TOUCHES_EDGE == 12
    assert MorpionEntityRelationType.MOVE_WINDOW_CONTAINS_EDGE == 13
    assert MorpionEntityRelationType.EDGE_IN_MOVE_WINDOW == 14
    assert MorpionEntityRelationType.MOVES_SHARE_NEW_DOT == 15


def test_relational_converter_contract_and_determinism() -> None:
    """Conversion should be typed, bounded, unique, sorted, and deterministic."""
    state = _make_one_step_state()
    converter = MorpionRelationalEntityTokenConverter(max_tokens=512)

    first = converter.state_to_tensors(state)
    second = converter.state_to_tensors(state)
    relation_rows = first.relation_triples.tolist()

    assert first.token_tensor.dtype == torch.float32
    assert first.token_tensor.shape[1] == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert first.relation_triples.dtype == torch.long
    assert first.relation_triples.ndim == 2
    assert first.relation_triples.shape[1] == 3
    assert relation_rows == sorted(relation_rows)
    assert len(relation_rows) == len({tuple(row) for row in relation_rows})
    assert int(first.relation_triples[:, 2].min().item()) >= 1
    assert int(first.relation_triples[:, 2].max().item()) < 16
    assert int(first.relation_triples[:, :2].min().item()) >= 0
    assert int(first.relation_triples[:, :2].max().item()) < first.token_tensor.shape[0]
    assert torch.equal(first.token_tensor, second.token_tensor)
    assert torch.equal(first.relation_triples, second.relation_triples)
    model_inputs = converter.state_to_model_input_tensors(state)
    assert torch.equal(model_inputs[0], first.token_tensor)
    assert torch.equal(model_inputs[1], first.relation_triples)


def test_move_dot_slot_relations_follow_action_window_order() -> None:
    """MOVE-to-DOT relation IDs should encode raw five-window slot order."""
    state = _make_standard_state()
    layout = MorpionEntityTokenConverter(max_tokens=512).state_to_layout(state)
    relations = _relation_set(
        MorpionRelationalEntityTokenConverter(max_tokens=512)
        .state_to_tensors(state)
        .relation_triples
    )
    action, move_index = next(iter(layout.move_index_by_action.items()))
    direction_index, x0, y0, _missing_index = action
    dx, dy = DIRECTIONS[direction_index]
    points = tuple((x0 + slot * dx, y0 + slot * dy) for slot in range(5))

    for slot, point in enumerate(points):
        dot_index = layout.dot_index_by_point.get(point)
        if dot_index is None:
            continue
        assert (
            move_index,
            dot_index,
            int(MorpionEntityRelationType.MOVE_TO_DOT_SLOT_0) + slot,
        ) in relations
        assert (
            dot_index,
            move_index,
            int(MorpionEntityRelationType.DOT_SLOT_0_OF_MOVE) + slot,
        ) in relations
    assert len(
        {
            int(MorpionEntityRelationType.MOVE_TO_DOT_SLOT_0) + slot
            for slot in range(5)
        }
    ) == 5


def test_move_edge_relations_use_canonical_segment_identity() -> None:
    """MOVE-window edge lookup should tolerate reversed segment endpoints."""
    state = _make_one_step_state()
    segment = next(iter(state.used_unit_segments))
    point_a, point_b = canonical_segment(segment)
    direction = point_b[0] - point_a[0], point_b[1] - point_a[1]
    action = (DIRECTIONS.index(direction), point_a[0], point_a[1], 4)
    dynamics = cast("MorpionDynamics", _FixedActionDynamics(action))
    layout = MorpionEntityTokenConverter(
        dynamics=dynamics,
        max_tokens=512,
    ).state_to_layout(state)
    relations = _relation_set(
        MorpionRelationalEntityTokenConverter(
            dynamics=dynamics,
            max_tokens=512,
        )
        .state_to_tensors(state)
        .relation_triples
    )
    match: tuple[int, int] | None = None
    for action, move_index in layout.move_index_by_action.items():
        direction_index, x0, y0, _missing_index = action
        dx, dy = DIRECTIONS[direction_index]
        points = tuple((x0 + slot * dx, y0 + slot * dy) for slot in range(5))
        for slot in range(4):
            reversed_segment = points[slot + 1], points[slot]
            edge_index = layout.edge_index_by_segment.get(
                canonical_segment(reversed_segment)
            )
            if edge_index is not None:
                match = move_index, edge_index
                break
        if match is not None:
            break

    assert match is not None
    move_index, edge_index = match
    assert (
        move_index,
        edge_index,
        int(MorpionEntityRelationType.MOVE_WINDOW_CONTAINS_EDGE),
    ) in relations
    assert (
        edge_index,
        move_index,
        int(MorpionEntityRelationType.EDGE_IN_MOVE_WINDOW),
    ) in relations


def test_edge_endpoint_dot_relations_are_exact() -> None:
    """EDGE endpoint relations should include endpoints and exclude other dots."""
    state = _make_one_step_state()
    layout = MorpionEntityTokenConverter(max_tokens=512).state_to_layout(state)
    relations = _relation_set(
        MorpionRelationalEntityTokenConverter(max_tokens=512)
        .state_to_tensors(state)
        .relation_triples
    )
    segment, edge_index = next(iter(layout.edge_index_by_segment.items()))
    for point in segment:
        dot_index = layout.dot_index_by_point[point]
        assert (
            edge_index,
            dot_index,
            int(MorpionEntityRelationType.EDGE_TOUCHES_DOT),
        ) in relations
        assert (
            dot_index,
            edge_index,
            int(MorpionEntityRelationType.DOT_TOUCHES_EDGE),
        ) in relations

    non_endpoint_index = next(
        index
        for point, index in layout.dot_index_by_point.items()
        if point not in segment
    )
    assert (
        edge_index,
        non_endpoint_index,
        int(MorpionEntityRelationType.EDGE_TOUCHES_DOT),
    ) not in relations
    assert (
        non_endpoint_index,
        edge_index,
        int(MorpionEntityRelationType.DOT_TOUCHES_EDGE),
    ) not in relations


def test_moves_sharing_new_dot_have_two_directed_relations_without_self_edges() -> None:
    """Moves with one missing point should be linked in both directions only."""
    dynamics = MorpionDynamics()
    state = dynamics.step(
        _make_standard_state(),
        (0, -5, -2, 4),
    ).next_state
    layout = MorpionEntityTokenConverter(max_tokens=512).state_to_layout(state)
    groups: dict[tuple[int, int], list[int]] = {}
    for action, move_index in layout.move_index_by_action.items():
        direction_index, x0, y0, missing_slot = action
        dx, dy = DIRECTIONS[direction_index]
        missing_point = x0 + missing_slot * dx, y0 + missing_slot * dy
        groups.setdefault(missing_point, []).append(move_index)
    shared_indices = next(indices for indices in groups.values() if len(indices) >= 2)
    move_a, move_b = shared_indices[:2]
    relations = _relation_set(
        MorpionRelationalEntityTokenConverter(max_tokens=512)
        .state_to_tensors(state)
        .relation_triples
    )
    relation_type = int(MorpionEntityRelationType.MOVES_SHARE_NEW_DOT)

    assert (move_a, move_b, relation_type) in relations
    assert (move_b, move_a, relation_type) in relations
    assert (move_a, move_a, relation_type) not in relations
    assert (move_b, move_b, relation_type) not in relations


def test_relation_generation_is_safe_under_entity_truncation() -> None:
    """No relation should outlive either of its truncated entity tokens."""
    state = _make_one_step_state()
    global_only = MorpionRelationalEntityTokenConverter(max_tokens=1).state_to_tensors(
        state
    )
    dots_only = MorpionRelationalEntityTokenConverter(max_tokens=5).state_to_tensors(
        state
    )

    assert global_only.token_tensor.shape == (1, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert global_only.relation_triples.shape == (0, 3)
    assert dots_only.relation_triples.shape == (0, 3)
    for max_tokens in (1, 5, 32, 64, 512):
        relational = MorpionRelationalEntityTokenConverter(
            max_tokens=max_tokens
        ).state_to_tensors(state)
        if relational.relation_triples.numel():
            assert int(relational.relation_triples[:, :2].min().item()) >= 0
            assert int(relational.relation_triples[:, :2].max().item()) < int(
                relational.token_tensor.shape[0]
            )


def test_relational_collation_pads_tokens_relations_and_targets() -> None:
    """Relational collation should preserve rows and zero-pad both inputs."""
    sample_a = MorpionRelationalEntityTokenSupervisedSample(
        input_tensor=torch.ones((2, MORPION_ENTITY_TOKEN_FEATURE_DIM)),
        auxiliary_input_tensors=(torch.tensor([[0, 1, 1]], dtype=torch.long),),
        target_tensor=torch.tensor([0.25]),
        is_batch=False,
    )
    sample_b = MorpionRelationalEntityTokenSupervisedSample(
        input_tensor=torch.full((4, MORPION_ENTITY_TOKEN_FEATURE_DIM), 2.0),
        auxiliary_input_tensors=(
            torch.tensor([[1, 2, 11], [2, 1, 12]], dtype=torch.long),
        ),
        target_tensor=torch.tensor([-0.5]),
        is_batch=False,
    )

    batch = collate_morpion_relational_entity_token_supervised_samples(
        (sample_a, sample_b)
    )
    relations = batch.auxiliary_input_tensors[0]

    assert batch.input_tensor.shape == (2, 4, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert relations.shape == (2, 2, 3)
    assert relations.dtype == torch.long
    assert batch.target_tensor.shape == (2, 1)
    assert len(batch.auxiliary_input_tensors) == 1
    assert torch.equal(batch.input_tensor[0, :2], sample_a.input_tensor)
    assert torch.all(batch.input_tensor[0, 2:] == 0)
    assert torch.equal(relations[0, 0], torch.tensor([0, 1, 1]))
    assert torch.equal(relations[0, 1], torch.tensor([0, 0, 0]))
    assert torch.equal(batch.target_tensor[:, 0], torch.tensor([0.25, -0.5]))


def test_relational_collation_empty_and_invalid_samples() -> None:
    """Empty shapes should be canonical and malformed auxiliaries rejected."""
    empty = collate_morpion_relational_entity_token_supervised_samples(())
    assert empty.input_tensor.shape == (0, 0, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert empty.auxiliary_input_tensors[0].shape == (0, 0, 3)
    assert empty.target_tensor.shape == (0, 1)

    base = {
        "input_tensor": torch.ones((2, MORPION_ENTITY_TOKEN_FEATURE_DIM)),
        "target_tensor": torch.tensor([0.0]),
        "is_batch": False,
    }
    for auxiliaries in (
        (),
        (torch.empty((0, 3), dtype=torch.long),) * 2,
        (torch.empty((2, 2), dtype=torch.long),),
    ):
        with pytest.raises(ValueError):
            collate_morpion_relational_entity_token_supervised_samples(
                (
                    MorpionRelationalEntityTokenSupervisedSample(
                        **base,
                        auxiliary_input_tensors=auxiliaries,
                    ),
                )
            )


def test_relational_row_conversion_and_eager_dataset(tmp_path: Path) -> None:
    """Rows and eager datasets should expose exact two-input conversion."""
    rows_path = _build_rows_file(tmp_path, target_values=(0.25, -0.5))
    rows = load_morpion_supervised_rows(rows_path).rows
    dynamics = MorpionDynamics()
    converter = MorpionRelationalEntityTokenConverter(
        dynamics=dynamics,
        max_tokens=128,
    )
    sample = process_morpion_supervised_row_to_relational_entity_token_tensors(
        rows[0],
        dynamics=dynamics,
        converter=converter,
    )
    state = dynamics.wrap_atomheart_state(
        decode_morpion_state_ref_payload(rows[0].state_ref_payload)
    )
    direct = converter.state_to_tensors(state)

    assert sample.input_tensor.ndim == 2
    assert sample.input_tensor.shape[1] == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert len(sample.auxiliary_input_tensors) == 1
    assert sample.auxiliary_input_tensors[0].shape[1] == 3
    assert sample.target_tensor.shape == (1,)
    assert sample.is_batch is False
    assert torch.equal(sample.input_tensor, direct.token_tensor)
    assert torch.equal(sample.auxiliary_input_tensors[0], direct.relation_triples)

    dataset = MorpionRelationalEntityTokenSupervisedDataset(
        MorpionRelationalEntityTokenSupervisedDatasetArgs(
            file_name=rows_path,
            max_tokens=128,
        )
    )
    assert len(dataset) == 2
    assert dataset.input_dim == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert dataset.feature_names() == MORPION_ENTITY_TOKEN_FEATURE_NAMES
    assert torch.equal(dataset[0].input_tensor, sample.input_tensor)


def test_relational_batch_can_call_a_two_input_model() -> None:
    """Collated relation tensors should be directly unpackable into a model."""

    class RelationShapeModel(torch.nn.Module):
        """Minimal model asserting the relational batch rank contract."""

        def forward(
            self,
            tokens: torch.Tensor,
            relations: torch.Tensor,
        ) -> torch.Tensor:
            """Return one scalar per batch after checking both input ranks."""
            assert tokens.ndim == 3
            assert relations.ndim == 3
            return tokens[..., 0].mean(dim=1, keepdim=True)

    state = _make_standard_state()
    relational = MorpionRelationalEntityTokenConverter(max_tokens=128).state_to_tensors(
        state
    )
    sample = MorpionRelationalEntityTokenSupervisedSample(
        input_tensor=relational.token_tensor,
        auxiliary_input_tensors=(relational.relation_triples,),
        target_tensor=torch.tensor([0.0]),
        is_batch=False,
    )
    batch = collate_morpion_relational_entity_token_supervised_samples((sample,))

    output = RelationShapeModel()(*batch.get_model_input_tensors())

    assert output.shape == (1, 1)


def _relation_set(relation_triples: torch.Tensor) -> set[tuple[int, int, int]]:
    """Return integer relation rows as a set for exact membership assertions."""
    return {tuple(row) for row in relation_triples.tolist()}
