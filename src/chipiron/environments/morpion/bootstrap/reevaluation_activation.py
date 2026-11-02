"""Recognize committed model replacement without relaxing proposal integrity."""

from typing import TYPE_CHECKING, Any

from .generation_publication_io import (
    PUBLICATION_KEY,
    committed_evaluator,
    publication_records,
    read_committed_descriptor,
    require,
)
from .reevaluation_transaction import transaction_state

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState


class ReevaluationModelActivatedError(Exception):
    """An unpublished job was superseded by a verified committed activation."""

    def __init__(self, model: dict[str, Any], generation: int, boundary: str) -> None:
        """Retain the captured identity for the discarded-job result and timing."""
        super().__init__("Committed activation superseded unpublished Reevaluation")
        self.model = model
        self.generation = generation
        self.boundary = boundary


def raise_for_committed_activation(
    paths: MorpionBootstrapPaths,
    before: MorpionBootstrapRunState | None,
    after: MorpionBootstrapRunState,
    *,
    boundary: str,
) -> None:
    """Raise a benign outcome only for a complete, validated M1→M2 chain.

    Unexpected model edits, foreign epochs, corrupt payloads and broken activation
    receipts remain errors. Reading old sealed descriptors needs no retired payload.
    This function never publishes, repairs, acknowledges or advances a cursor.
    """
    if before is None or any(
        state.metadata.get(PUBLICATION_KEY, {}).get("evaluator_contract") != 1
        for state in (before, after)
    ):
        return
    previous_head = publication_records(before.metadata)[-1]
    records = publication_records(after.metadata)
    position = next(
        (i for i, record in enumerate(records) if record["id"] == previous_head["id"]),
        None,
    )
    require(position is not None, "Reevaluation authority changed its ancestry.")
    assert position is not None
    require(
        records[position]["descriptor_sha256"] == previous_head["descriptor_sha256"],
        "Reevaluation captured descriptor changed.",
    )
    old_model = read_committed_descriptor(paths.work_dir, previous_head)[
        "active_evaluator"
    ]
    new_model, _ = committed_evaluator(paths.work_dir, after)
    if old_model == new_model:
        return
    old_transaction, new_transaction = (
        transaction_state(before),
        transaction_state(after),
    )
    require(
        old_transaction is not None
        and new_transaction is not None
        and {k: v for k, v in old_transaction.items() if k != "accepted_cursor"}
        == {k: v for k, v in new_transaction.items() if k != "accepted_cursor"},
        "Reevaluation activation changed workspace/epoch/producer identity.",
    )
    from .model_activation import activation_completion

    model = old_model
    transitions = 0
    for record in records[position + 1 :]:
        if "activation_request_sha256" not in record:
            continue
        completion = activation_completion(paths, after, record)
        request = completion["request"]
        require(request["previous_model"] == model, "Broken model activation chain.")
        model = request["candidate"]
        transitions += 1
    require(
        transitions > 0 and model == new_model, "Unproven Reevaluation model change."
    )
    raise ReevaluationModelActivatedError(old_model, before.generation, boundary)
