"""Small immutable-result I/O shared by Dataset and Training.

No stage dispatch, journal or authority: callers validate their own contracts and
hold existing Growth ownership before attaching a reference to run_state.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .derived.provenance import inside
from .generation_publication_io import (
    PUBLICATION_KEY,
    file_digest,
    publication_records,
    read_object,
    require,
)
from .reevaluation_transaction import transaction_state

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState


def publication_record(
    state: MorpionBootstrapRunState, generation: int
) -> dict[str, Any]:
    """Require a unique committed scientific generation."""
    records = [
        r for r in publication_records(state.metadata) if r["generation"] == generation
    ]
    require(len(records) == 1, "Worker result source is not a committed generation.")
    return records[0]


def workspace_epoch(state: MorpionBootstrapRunState) -> dict[str, str]:
    """Read the existing authoritative workspace and epoch."""
    data = transaction_state(state)
    require(
        data is not None, "Worker result requires an authoritative workspace/epoch."
    )
    assert data is not None
    return {key: data[key] for key in ("workspace_id", "epoch_id")}


def attempt_directory(
    paths: MorpionBootstrapPaths,
    attempt_id: str,
    *,
    namespace: str,
    private: bool = False,
) -> Path:
    """Derive a private or immutable path from a strict attempt identity."""
    require(
        isinstance(attempt_id, str)
        and len(attempt_id) == 32
        and all(c in "0123456789abcdef" for c in attempt_id),
        "Malformed Worker result attempt identity.",
    )
    return (
        paths.work_dir / namespace / ("staging" if private else "objects") / attempt_id
    )


def read_proposal(
    paths: MorpionBootstrapPaths, reference: dict[str, Any], *, namespace: str
) -> dict[str, Any]:
    """Read a proposal only after checking its exact descriptor digest."""
    require(
        set(reference) == {"attempt_id", "descriptor_sha256"},
        "Malformed accepted Worker result reference.",
    )
    directory = attempt_directory(paths, reference["attempt_id"], namespace=namespace)
    path = inside(
        paths.work_dir, paths.relative_to_work_dir(directory / "proposal.json")
    )
    require(
        file_digest(path) == reference["descriptor_sha256"],
        "Worker result descriptor changed.",
    )
    proposal = read_object(path)
    require(
        proposal["admission"]["attempt_id"] == reference["attempt_id"],
        "Worker result attempt identity mismatch.",
    )
    return proposal


def with_accepted_result(
    state: MorpionBootstrapRunState,
    generation: int,
    key: str,
    reference: dict[str, Any],
) -> MorpionBootstrapRunState:
    """Attach one reference to an existing entry; never create another history."""
    record = publication_record(state, generation)
    require(record.get(key) in (None, reference), "Conflicting accepted worker result.")
    publication = state.metadata[PUBLICATION_KEY]
    return replace(
        state,
        metadata={
            **state.metadata,
            PUBLICATION_KEY: {
                **publication,
                "generations": [
                    {**r, key: reference} if r["generation"] == generation else r
                    for r in publication["generations"]
                ],
            },
        },
    )


def result_protocol_enabled(
    state: MorpionBootstrapRunState, key: str, accepted: str
) -> bool:
    """Validate an explicit v1 result marker and prevent silent protocol downgrade."""
    if key not in state.metadata:
        require(
            not any(
                accepted in r
                for r in state.metadata.get(PUBLICATION_KEY, {}).get("generations", [])
            ),
            "Accepted worker references require their versioned protocol.",
        )
        return False
    marker = state.metadata[key]
    require(
        marker == {"schema_version": 1}
        and isinstance(marker.get("schema_version"), int)
        and not isinstance(marker["schema_version"], bool)
        and transaction_state(state) is not None,
        "Unsupported worker result protocol or missing workspace/epoch.",
    )
    return True
