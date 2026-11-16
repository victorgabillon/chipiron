"""One accepted producer cursor in run_state; immutable proposals are not commits.

Only explicit fresh closed-snapshot genesis enables this protocol. Epoch identity
fences Reevaluation proposals, not the independent Dataset/Training writers.
"""

import hashlib
import json
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from .generation_publication_io import RETIRED_PAYLOAD_KEY, publication_records, require
from .pipeline_artifacts import (
    MorpionReevaluationCursor,
    MorpionReevaluationPatch,
    reevaluation_cursor_from_dict,
    reevaluation_cursor_to_dict,
)

if TYPE_CHECKING:
    from .run_state import MorpionBootstrapRunState

TRANSACTION_KEY = "atomic_reevaluation"
PROTOCOL = "morpion_reevaluation_proposal"
VERSION = 1
RAW_NUMERIC_POLICY = "raw_nonterminal_v1"
GROWTH_NUMERIC_POLICY = "anemone_depth_scaled_nonterminal_v1"


def cursor_digest(cursor: object) -> str:
    """Identify the complete accepted continuation, including explicit genesis null."""
    return hashlib.sha256(
        json.dumps(
            cursor, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def initial_transaction(
    *, growth_scoring: bool = False, rolling: bool = False
) -> dict[str, Any]:
    """Create identities only for a new workspace's existing durable genesis commit."""
    require(not rolling or growth_scoring, "Rolling traversal requires Growth scoring.")
    return {
        "protocol": PROTOCOL,
        "schema_version": 3 if rolling else 2 if growth_scoring else VERSION,
        **({"numeric_policy": GROWTH_NUMERIC_POLICY} if growth_scoring else {}),
        "workspace_id": uuid.uuid4().hex,
        "epoch_id": uuid.uuid4().hex,
        "producer_id": uuid.uuid4().hex,
        "accepted_cursor": None,
    }


def complete_cursor(data: object) -> MorpionReevaluationCursor:
    """Reject missing, unknown or normalized fields instead of inventing continuation."""
    cursor = reevaluation_cursor_from_dict(data)
    require(
        isinstance(data, dict)
        and set(data) == set(reevaluation_cursor_to_dict(cursor))
        and cursor_digest(data) == cursor_digest(reevaluation_cursor_to_dict(cursor))
        and all(
            isinstance(data[key], int) and not isinstance(data[key], bool)
            for key in (
                "completed_full_pass_count",
                "tree_generation",
                "evaluator_generation",
            )
        )
        and cursor.last_patch_id is not None,
        "Incomplete or malformed accepted cursor.",
    )
    return cursor


def transaction_state(state: MorpionBootstrapRunState) -> dict[str, Any] | None:
    """Dispatch explicitly; never fall back from malformed current protocol data."""
    if TRANSACTION_KEY not in state.metadata:
        return None
    data = state.metadata[TRANSACTION_KEY]
    require(
        isinstance(data, dict)
        and set(data)
        == {
            "protocol",
            "schema_version",
            "workspace_id",
            "epoch_id",
            "producer_id",
            "accepted_cursor",
        }
        | ({"numeric_policy"} if data.get("schema_version") in (2, 3) else set())
        and data["protocol"] == PROTOCOL
        and isinstance(data["schema_version"], int)
        and not isinstance(data["schema_version"], bool)
        and data["schema_version"] in (VERSION, 2, 3)
        and (
            data["schema_version"] == VERSION
            or data.get("numeric_policy") == GROWTH_NUMERIC_POLICY
        ),
        "Unsupported atomic Reevaluation protocol.",
    )
    assert isinstance(data, dict)
    for key in ("workspace_id", "epoch_id", "producer_id"):
        value = data[key]
        require(
            isinstance(value, str)
            and len(value) == 32
            and all(c in "0123456789abcdef" for c in value),
            "Malformed Reevaluation workspace/epoch/producer identity.",
        )
    if data["accepted_cursor"] is not None:
        complete_cursor(data["accepted_cursor"])
    return data


def source_identity(
    state: MorpionBootstrapRunState,
    generation: int | None,
    *,
    require_payload: bool = False,
) -> dict[str, Any]:
    """Bind the exact committed seal and science, without another inventory."""
    records = [
        r for r in publication_records(state.metadata) if r["generation"] == generation
    ]
    require(
        len(records) == 1 and records[0].get("publication_version") == 3,
        "Atomic Reevaluation requires a committed closed scientific source.",
    )
    record = records[0]
    require(
        not require_payload or RETIRED_PAYLOAD_KEY not in record,
        "Retired scientific source cannot admit a new proposal.",
    )
    return {
        key: record[key] for key in ("id", "descriptor_sha256", "scientific_identity")
    }


def proposal_patch(
    patch: MorpionReevaluationPatch,
    captured: dict[str, Any],
    source: dict[str, Any],
    model: dict[str, Any],
    next_cursor: MorpionReevaluationCursor,
) -> MorpionReevaluationPatch:
    """Bind identities captured BEFORE computation; no destination identity lookup."""
    envelope = {
        key: value for key, value in captured.items() if key != "accepted_cursor"
    }
    envelope.update(
        source=source,
        model=model,
        expected_cursor_sha256=cursor_digest(captured["accepted_cursor"]),
        next_cursor=reevaluation_cursor_to_dict(next_cursor),
    )
    return replace(patch, metadata={**patch.metadata, TRANSACTION_KEY: envelope})


def validate_proposal(
    state: MorpionBootstrapRunState,
    patch: MorpionReevaluationPatch,
    *,
    duplicate_cursor_digest: str | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate identity before dedup; only NEW proposals compare the current cursor.

    Exact byte equality with a receipt is independently required by the consumer.
    A duplicate's next cursor is evidence only, never a command to rewind.
    """
    data = transaction_state(state)
    require(data is not None, "Atomic proposal requires an initialized authority.")
    assert data is not None
    envelope = patch.metadata.get(TRANSACTION_KEY)
    identity = {k: v for k, v in data.items() if k != "accepted_cursor"}
    require(
        isinstance(envelope, dict)
        and set(envelope)
        == set(identity) | {"source", "model", "expected_cursor_sha256", "next_cursor"}
        and all(
            type(envelope[k]) is type(v) and envelope[k] == v
            for k, v in identity.items()
        ),
        "Proposal protocol/workspace/epoch/producer mismatch.",
    )
    assert isinstance(envelope, dict)
    require(
        envelope["source"]
        == source_identity(
            state,
            patch.tree_generation,
            require_payload=duplicate_cursor_digest is None,
        ),
        "Proposal scientific source identity mismatch.",
    )
    if data.get("numeric_policy") == GROWTH_NUMERIC_POLICY:
        for row in patch.rows:
            source = row.metadata.get("source")
            require(
                source
                in (
                    "active_model_reevaluation",
                    "terminal_existing_value",
                    "snapshot_existing_values",
                ),
                "Undeclared numeric source in scaled proposal.",
            )
            if source == "active_model_reevaluation":
                depth = row.metadata.get("tree_depth")
                require(
                    row.metadata.get("numeric_policy") == GROWTH_NUMERIC_POLICY
                    and isinstance(depth, int)
                    and not isinstance(depth, bool)
                    and depth >= 0
                    and row.is_exact is False
                    and row.is_terminal is False,
                    "Fresh estimate has incompatible numeric policy or certainty.",
                )
    next_cursor = complete_cursor(envelope["next_cursor"])
    require(
        (
            next_cursor.last_patch_id,
            next_cursor.tree_generation,
            next_cursor.evaluator_generation,
            next_cursor.evaluator_name,
            next_cursor.model_bundle_path,
            next_cursor.updated_at_utc,
        )
        == (
            patch.patch_id,
            patch.tree_generation,
            patch.evaluator_generation,
            patch.evaluator_name,
            patch.model_bundle_path,
            patch.created_at_utc,
        )
        and next_cursor.next_node_cursor == patch.metadata.get("next_node_cursor")
        and isinstance(patch.metadata.get("completed_full_pass"), bool),
        "Proposal cursor does not describe this patch.",
    )
    if duplicate_cursor_digest is not None:
        require(
            cursor_digest(envelope["next_cursor"]) == duplicate_cursor_digest,
            "Committed proposal cursor evidence mismatch.",
        )
        return envelope
    require(
        envelope["expected_cursor_sha256"] == cursor_digest(data["accepted_cursor"]),
        "Stale expected accepted cursor.",
    )
    previous = (
        None
        if data["accepted_cursor"] is None
        else complete_cursor(data["accepted_cursor"])
    )
    rolling = data["schema_version"] == 3
    if rolling:
        _validate_rolling_prefix(state, patch, previous, next_cursor, work_dir)
    same_model = previous is not None and (
        previous.evaluator_generation,
        previous.evaluator_name,
        previous.model_bundle_path,
    ) == (patch.evaluator_generation, patch.evaluator_name, patch.model_bundle_path)
    prior_passes = (
        previous.completed_full_pass_count
        if previous is not None and (same_model or rolling)
        else 0
    )
    require(
        next_cursor.completed_full_pass_count
        == prior_passes + int(patch.metadata["completed_full_pass"] is True),
        "Proposal completed-pass counter mismatch.",
    )
    return envelope


def rolling_source_generation(state: MorpionBootstrapRunState) -> int | None:
    """Return the durable incomplete-pass root, independent of the current model."""
    data = transaction_state(state)
    if data is None or data["schema_version"] != 3 or data["accepted_cursor"] is None:
        return None
    cursor = complete_cursor(data["accepted_cursor"])
    require(
        cursor.metadata.get("rolling_source")
        == source_identity(state, cursor.tree_generation),
        "Rolling cursor source identity changed.",
    )
    if cursor.next_node_cursor is None:
        return None
    source_identity(state, cursor.tree_generation, require_payload=True)
    return cursor.tree_generation


def _validate_rolling_prefix(
    state: MorpionBootstrapRunState,
    patch: MorpionReevaluationPatch,
    previous: MorpionReevaluationCursor | None,
    next_cursor: MorpionReevaluationCursor,
    work_dir: Path | None,
) -> None:
    """Authenticate the exact finite inventory prefix before native application.

    The existing consumer supplies its own workspace; no producer-chosen path is
    trusted. Exact receipt retries use the original digest and need no old payload.
    """
    from .derived.provenance import inside
    from .sharded_training_reader import ShardedTrainingGenerationView

    require(work_dir is not None, "Rolling prefix validation requires owned workspace.")
    assert work_dir is not None
    frozen = rolling_source_generation(state)
    require(
        frozen is None or frozen == patch.tree_generation,
        "Rolling pass source changed.",
    )
    if frozen is None and previous is not None:
        require(
            patch.tree_generation is not None
            and previous.tree_generation is not None
            and patch.tree_generation > previous.tree_generation,
            "A new rolling pass requires a later committed inventory.",
        )
    source = source_identity(state, patch.tree_generation, require_payload=True)
    require(
        next_cursor.metadata
        == {"source": "reevaluation_worker", "rolling_source": source},
        "Rolling next cursor source mismatch.",
    )
    record = next(
        r
        for r in publication_records(state.metadata)
        if r["generation"] == patch.tree_generation
    )
    limit = patch.metadata.get("max_nodes_per_patch")
    require(
        isinstance(limit, int) and not isinstance(limit, bool) and limit > 0,
        "Rolling batches require positive capacity.",
    )
    assert isinstance(limit, int)
    start = (
        previous.next_node_cursor
        if frozen is not None and previous is not None
        else None
    )
    with ShardedTrainingGenerationView(
        inside(work_dir, record["tree_snapshot_path"]),
        allow_scientific_v2=True,
        allow_closed_snapshot=True,
    ) as view:
        ids, continuation, finished = view.select_finite_node_window(start, limit)
        view.validate_source_identity()
    require(
        bool(ids) and tuple(row.node_id for row in patch.rows) == ids,
        "Rolling proposal must consume the exact nonempty inventory prefix.",
    )
    require(
        (patch.start_cursor, patch.end_cursor) == (ids[0], ids[-1])
        and next_cursor.next_node_cursor == continuation
        and patch.metadata["completed_full_pass"] is finished,
        "Rolling proposal continuation skipped or repeated inventory.",
    )
