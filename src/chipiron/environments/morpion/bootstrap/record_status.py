"""Morpion-specific bootstrap record semantics and snapshot extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from chipiron.environments.morpion.learning import (
    InvalidMorpionStateRefPayloadError,
    decode_morpion_state_ref_payload,
)

MORPION_BOOTSTRAP_GAME = "morpion"
MORPION_BOOTSTRAP_VARIANT = "5T"
MORPION_BOOTSTRAP_INITIAL_PATTERN = "greek_cross"
MORPION_BOOTSTRAP_INITIAL_POINT_COUNT = 36

if TYPE_CHECKING:
    from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRecordStatus:
    """Structured Morpion record semantics for one bootstrap cycle."""

    variant: str | None
    initial_pattern: str | None
    initial_point_count: int | None
    current_best_moves_since_start: int | None
    current_best_total_points: int | None
    current_best_is_exact: bool | None
    current_best_source: str | None


def morpion_bootstrap_experiment_metadata() -> dict[str, object]:
    """Return the canonical metadata for the current Morpion bootstrap run."""
    return {
        "game": MORPION_BOOTSTRAP_GAME,
        "variant": MORPION_BOOTSTRAP_VARIANT,
        "initial_pattern": MORPION_BOOTSTRAP_INITIAL_PATTERN,
        "initial_point_count": MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    }


def default_morpion_record_status() -> MorpionBootstrapRecordStatus:
    """Return the default empty record status for the current experiment."""
    return MorpionBootstrapRecordStatus(
        variant=MORPION_BOOTSTRAP_VARIANT,
        initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN,
        initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
        current_best_moves_since_start=None,
        current_best_total_points=None,
        current_best_is_exact=None,
        current_best_source=None,
    )


def carried_forward_morpion_record_status(
    previous: MorpionBootstrapRecordStatus | None,
) -> MorpionBootstrapRecordStatus:
    """Return the record status that should appear on one no-save cycle event."""
    if previous is None:
        return default_morpion_record_status()
    normalized_previous = _normalized_record_status(previous)
    return replace(
        normalized_previous,
        current_best_source="carried_from_run_state",
    )


def extract_morpion_record_status_from_training_tree_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    variant: str = MORPION_BOOTSTRAP_VARIANT,
    initial_pattern: str = MORPION_BOOTSTRAP_INITIAL_PATTERN,
    initial_point_count: int = MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
) -> MorpionBootstrapRecordStatus:
    """Extract the best known Morpion achievement from one training snapshot."""
    best_status = MorpionBootstrapRecordStatus(
        variant=variant,
        initial_pattern=initial_pattern,
        initial_point_count=initial_point_count,
        current_best_moves_since_start=None,
        current_best_total_points=None,
        current_best_is_exact=None,
        current_best_source=None,
    )
    best_rank: tuple[int, int] | None = None

    for node in snapshot.nodes:
        if node.state_ref_payload is None:
            continue
        try:
            decoded_state = decode_morpion_state_ref_payload(node.state_ref_payload)
        except InvalidMorpionStateRefPayloadError:
            continue
        if decoded_state.variant.value != variant:
            continue

        moves_since_start = int(decoded_state.moves)
        candidate_rank = (moves_since_start, 1 if node.is_exact else 0)
        if best_rank is not None and candidate_rank <= best_rank:
            continue

        best_rank = candidate_rank
        best_status = MorpionBootstrapRecordStatus(
            variant=variant,
            initial_pattern=initial_pattern,
            initial_point_count=initial_point_count,
            current_best_moves_since_start=moves_since_start,
            current_best_total_points=initial_point_count + moves_since_start,
            current_best_is_exact=node.is_exact,
            current_best_source=_snapshot_source_for_node(node),
        )

    return best_status


def resolve_record_status_for_cycle(
    *,
    snapshot: TrainingTreeSnapshot | None,
    previous_record_status: MorpionBootstrapRecordStatus | None,
) -> MorpionBootstrapRecordStatus:
    """Resolve the record status that should be written for one bootstrap cycle."""
    if snapshot is not None:
        return extract_morpion_record_status_from_training_tree_snapshot(snapshot)
    return carried_forward_morpion_record_status(previous_record_status)


def _normalized_record_status(
    status: MorpionBootstrapRecordStatus,
) -> MorpionBootstrapRecordStatus:
    """Fill missing experiment identity fields without changing known record data."""
    return MorpionBootstrapRecordStatus(
        variant=MORPION_BOOTSTRAP_VARIANT if status.variant is None else status.variant,
        initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN
        if status.initial_pattern is None
        else status.initial_pattern,
        initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT
        if status.initial_point_count is None
        else status.initial_point_count,
        current_best_moves_since_start=status.current_best_moves_since_start,
        current_best_total_points=status.current_best_total_points,
        current_best_is_exact=status.current_best_is_exact,
        current_best_source=status.current_best_source,
    )


def _snapshot_source_for_node(node: TrainingNodeSnapshot) -> str:
    """Return the canonical source label for one snapshot-derived best node."""
    if node.is_exact:
        return "snapshot_exact_node"
    if node.is_terminal:
        return "snapshot_terminal_node"
    return "snapshot_nonterminal_node"


__all__ = [
    "MORPION_BOOTSTRAP_GAME",
    "MORPION_BOOTSTRAP_INITIAL_PATTERN",
    "MORPION_BOOTSTRAP_INITIAL_POINT_COUNT",
    "MORPION_BOOTSTRAP_VARIANT",
    "MorpionBootstrapRecordStatus",
    "carried_forward_morpion_record_status",
    "default_morpion_record_status",
    "extract_morpion_record_status_from_training_tree_snapshot",
    "morpion_bootstrap_experiment_metadata",
    "resolve_record_status_for_cycle",
]
