"""Focused tests for checkpoint payload serialization helpers."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import anemone.checkpoints.build as checkpoint_build_module
import anemone.checkpoints.build_atoms as checkpoint_build_atoms_module
import anemone.checkpoints.build_context as checkpoint_build_context_module
import anemone.checkpoints.build_values as checkpoint_build_values_module
from anemone.checkpoints import (
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
    LinooSelectorCheckpointPayload,
    LinkedChildCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    TreeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
    checkpoint_payload_to_jsonable,
)
from valanga.evaluations import Certainty, Value


def test_checkpoint_payload_to_jsonable_matches_dataclasses_asdict() -> None:
    """The custom JSONable conversion should preserve checkpoint payload shape."""
    payload = SearchRuntimeCheckpointPayload(
        evaluator_version=7,
        tree=TreeCheckpointPayload(
            root_node_id=1,
            nodes=[
                AlgorithmNodeCheckpointPayload(
                    node_id=1,
                    parent_node_id=None,
                    branch_from_parent=None,
                    depth=0,
                    state_payload=AnchorCheckpointStatePayload(
                        anchor_ref={
                            "board": [0, 1, 2],
                            "metadata": {"phase": "opening", "turn": 4},
                        },
                    ),
                    generated_all_branches=True,
                    unopened_branches=[0, 1],
                    linked_children=[
                        LinkedChildCheckpointPayload(branch_key=1, child_node_id=2)
                    ],
                ),
                AlgorithmNodeCheckpointPayload(
                    node_id=2,
                    parent_node_id=1,
                    branch_from_parent={"type": "tuple", "items": [1, 2]},
                    depth=1,
                    state_payload=DeltaCheckpointStatePayload(
                        state_parent_node_id=1,
                        state_parent_branch=1,
                        delta_ref={"move": [1, 2]},
                    ),
                    generated_all_branches=False,
                ),
            ],
        ),
        latest_tree_expansions=TreeExpansionsCheckpointPayload(
            expansions_with_node_creation=[
                TreeExpansionCheckpointPayload(
                    child_node_id=2,
                    parent_node_id=1,
                    branch_key=1,
                    creation_child_node=True,
                )
            ],
        ),
        selector_state=LinooSelectorCheckpointPayload(
            version=3,
            last_selected_node_id=2,
        ),
    )

    assert checkpoint_payload_to_jsonable(payload) == asdict(payload)


def test_checkpoint_build_branch_collection_matches_reference_order() -> None:
    """The optimized branch collection serializer must preserve reference order."""
    branches = [3, 1, 2, 10]
    context = checkpoint_build_module._CheckpointBuildContext(  # pylint: disable=protected-access
        metrics=checkpoint_build_module._CheckpointBuildMetrics()  # pylint: disable=protected-access
    )

    serialized = checkpoint_build_module._serialize_branch_collection(  # pylint: disable=protected-access
        branches,
        context=context,
    )
    reference = sorted(
        (
            checkpoint_build_atoms_module.serialize_checkpoint_atom(branch)
            for branch in branches
        ),
        key=repr,
    )

    assert serialized == reference


def test_checkpoint_build_atom_serialization_cache_reuses_hashable_atoms(
    monkeypatch,
) -> None:
    """Repeated equal hashable atoms should hit the build-local serialization cache."""
    serialize_call_count = 0

    def _spy_serialize_checkpoint_atom(value: object) -> object:
        nonlocal serialize_call_count
        serialize_call_count += 1
        return value

    monkeypatch.setattr(
        checkpoint_build_atoms_module,
        "serialize_checkpoint_atom",
        _spy_serialize_checkpoint_atom,
    )
    context = checkpoint_build_module._CheckpointBuildContext(  # pylint: disable=protected-access
        metrics=checkpoint_build_module._CheckpointBuildMetrics()  # pylint: disable=protected-access
    )

    serialized = checkpoint_build_module._serialize_branch_collection(  # pylint: disable=protected-access
        [7, 7, 7],
        context=context,
    )

    assert serialized == [7, 7, 7]
    assert serialize_call_count == 1
    assert context.metrics.atom_serialize_cache_hits == 2


def test_checkpoint_build_parent_branch_cache_preserves_stable_order() -> None:
    """Per-node parent branch caching must preserve branch ordering semantics."""
    parent_node = SimpleNamespace(id=11)
    context = checkpoint_build_module._CheckpointBuildContext(  # pylint: disable=protected-access
        metrics=checkpoint_build_module._CheckpointBuildMetrics()  # pylint: disable=protected-access
    )

    parent_branch_cache = checkpoint_build_module._serialize_parent_branches(  # pylint: disable=protected-access
        parent_node,
        [3, 1, 2],
        context=context,
    )

    assert list(parent_branch_cache.ordered_branches) == [1, 2, 3]
    assert list(parent_branch_cache.serialized_branches) == [1, 2, 3]


def test_checkpoint_build_value_serialization_cache_reuses_value_identity(
    monkeypatch,
) -> None:
    """Repeated serialization of the same Value object should hit the build-local cache."""
    validation_call_count = 0
    real_validate = checkpoint_build_values_module.canonical_value.validate_value_semantics

    def _spy_validate_value_semantics(value: Value) -> Value:
        nonlocal validation_call_count
        validation_call_count += 1
        return real_validate(value)

    monkeypatch.setattr(
        checkpoint_build_values_module.canonical_value,
        "validate_value_semantics",
        _spy_validate_value_semantics,
    )
    context = checkpoint_build_module._CheckpointBuildContext(  # pylint: disable=protected-access
        metrics=checkpoint_build_module._CheckpointBuildMetrics()  # pylint: disable=protected-access
    )
    value = Value(
        score=1.25,
        certainty=Certainty.ESTIMATE,
        over_event=None,
        line=[3, 1, 3],
    )

    first_payload = checkpoint_build_module._serialize_optional_value(  # pylint: disable=protected-access
        value,
        context=context,
        value_kind="direct",
    )
    second_payload = checkpoint_build_module._serialize_optional_value(  # pylint: disable=protected-access
        value,
        context=context,
        value_kind="backed_up",
    )

    assert first_payload == second_payload
    assert validation_call_count == 1
    assert context.metrics.serialize_value_cache_hits == 1
    assert context.metrics.serialize_value_cache_misses == 1


def test_checkpoint_build_detail_log_includes_new_metrics(monkeypatch) -> None:
    """Detailed checkpoint build logging should expose the new additive metrics."""
    messages: list[str] = []

    def _capture_info(message: str, *args: object) -> None:
        messages.append(message % args)

    monkeypatch.setattr(
        checkpoint_build_context_module.anemone_logger,
        "info",
        _capture_info,
    )
    metrics = checkpoint_build_module._CheckpointBuildMetrics(  # pylint: disable=protected-access
        node_evaluation_calls=4,
        node_evaluation_total_s=1.25,
        evaluation_payload_reuse_candidates_missing_version=3,
        evaluation_payload_reuse_blocked_missing_version=3,
        tree_evaluation_access_calls=4,
        tree_evaluation_access_s=0.125,
        direct_value_access_calls=4,
        direct_value_access_s=0.0625,
        direct_value_serialize_calls=3,
        direct_value_serialize_s=0.5,
        backed_up_value_access_calls=2,
        backed_up_value_access_s=0.03125,
        backed_up_value_serialize_calls=2,
        backed_up_value_serialize_s=0.25,
        serialize_value_calls=5,
        serialize_value_total_s=0.75,
        serialize_value_cache_hits=3,
        serialize_value_cache_misses=2,
        value_semantic_validation_calls=2,
        value_semantic_validation_s=0.125,
        principal_variation_calls=1,
        principal_variation_serialize_s=0.1,
        decision_ordering_calls=1,
        decision_ordering_serialize_s=0.2,
        branch_frontier_calls=2,
        branch_frontier_total_s=0.3,
        backup_runtime_calls=2,
        backup_runtime_total_s=0.15,
        branch_collection_calls=5,
        branch_collection_total_s=0.7,
        branch_collection_sort_calls=5,
        branch_collection_sort_s=0.4,
        representative_parent_calls=4,
        representative_parent_total_s=0.9,
        reusable_parent_scan_calls=3,
        reusable_parent_scan_s=0.6,
        stored_or_first_branch_calls=2,
        stored_or_first_branch_total_s=0.2,
        first_branch_calls=2,
        first_branch_total_s=0.1,
        linked_children_calls=4,
        linked_children_total_s=0.8,
        linked_children_sort_calls=4,
        linked_children_sort_s=0.5,
        evaluation_atom_serialize_calls=8,
        evaluation_atom_serialize_total_s=0.4,
        evaluation_atom_serialize_cache_hits=6,
        evaluation_atom_serialize_cache_misses=2,
        atom_serialize_calls=10,
        atom_serialize_total_s=1.1,
        atom_serialize_cache_hits=6,
        atom_serialize_cache_misses=4,
        delta_payloads_emitted=1,
        anchor_payloads_emitted=1,
        node_count=2,
    )

    checkpoint_build_context_module._log_checkpoint_build_metrics(metrics)  # pylint: disable=protected-access

    assert any("[checkpoint-build-detail]" in message for message in messages)
    assert any(
        "evaluation_payload_reuse_candidates_missing_version=" in message
        for message in messages
    )
    assert any(
        "evaluation_payload_reuse_blocked_missing_version=" in message
        for message in messages
    )
    assert any("tree_evaluation_access_s=" in message for message in messages)
    assert any("serialize_value_calls=" in message for message in messages)
    assert any("value_semantic_validation_calls=" in message for message in messages)
    assert any("backup_runtime_total_s=" in message for message in messages)
    assert any("evaluation_atom_serialize_calls=" in message for message in messages)
    assert any("branch_collection_sort_s=" in message for message in messages)
    assert any("atom_serialize_calls=" in message for message in messages)
