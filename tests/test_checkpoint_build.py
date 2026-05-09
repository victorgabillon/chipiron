"""Focused tests for checkpoint payload serialization helpers."""

from __future__ import annotations

from dataclasses import asdict

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