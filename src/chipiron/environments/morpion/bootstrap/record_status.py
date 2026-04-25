"""Morpion-specific bootstrap record semantics and certified leaderboard helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chipiron.environments.morpion.learning import (
    InvalidMorpionStateRefPayloadError,
    decode_morpion_state_ref_payload,
)

MORPION_BOOTSTRAP_GAME = "morpion"
MORPION_BOOTSTRAP_VARIANT = "5T"
MORPION_BOOTSTRAP_INITIAL_PATTERN = "greek_cross"
MORPION_BOOTSTRAP_INITIAL_POINT_COUNT = 36
MORPION_LEADERBOARD_LIMIT_PER_VARIANT = 100

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRecordStatus:
    """Strict certified Morpion record semantics for one bootstrap cycle."""

    variant: str | None
    initial_pattern: str | None
    initial_point_count: int | None
    current_best_moves_since_start: int | None
    current_best_total_points: int | None
    current_best_is_exact: bool | None
    current_best_is_terminal: bool | None
    current_best_source: str | None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapFrontierStatus:
    """Best permissive frontier/debug status for one bootstrap cycle."""

    variant: str | None
    initial_pattern: str | None
    initial_point_count: int | None
    current_best_moves_since_start: int | None
    current_best_total_points: int | None
    current_best_is_exact: bool | None
    current_best_is_terminal: bool | None
    current_best_source: str | None


@dataclass(frozen=True, slots=True)
class MorpionCertifiedRecordCandidate:
    """Snapshot-derived certified Morpion terminal/exact candidate."""

    node_id: str
    variant: str
    initial_pattern: str
    initial_point_count: int
    moves_since_start: int
    total_points: int
    state_ref_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class MorpionFrontierNodeCandidate:
    """Cheap metadata-only Morpion frontier/status candidate."""

    node_id: str
    moves_since_start: int
    total_points: int
    is_terminal: bool
    is_exact: bool
    source: str


@dataclass(frozen=True, slots=True)
class MorpionFrontierResolution:
    """Resolved frontier status plus metadata-only candidate count."""

    status: MorpionBootstrapFrontierStatus
    candidate_count: int


@dataclass(frozen=True, slots=True)
class MorpionLeaderboardEntry:
    """One persisted certified Morpion leaderboard entry."""

    variant: str
    total_points: int
    moves_since_start: int
    is_terminal: bool
    is_exact: bool
    source: str
    state_fingerprint: str
    state_ref_payload: dict[str, object]
    run_work_dir: str
    generation: int
    cycle_index: int
    timestamp_utc: str


def morpion_bootstrap_experiment_metadata() -> dict[str, object]:
    """Return the canonical metadata for the current Morpion bootstrap run."""
    return {
        "game": MORPION_BOOTSTRAP_GAME,
        "variant": MORPION_BOOTSTRAP_VARIANT,
        "initial_pattern": MORPION_BOOTSTRAP_INITIAL_PATTERN,
        "initial_point_count": MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    }


def default_morpion_record_status() -> MorpionBootstrapRecordStatus:
    """Return the default empty certified record status for the experiment."""
    return MorpionBootstrapRecordStatus(
        variant=MORPION_BOOTSTRAP_VARIANT,
        initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN,
        initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
        current_best_moves_since_start=None,
        current_best_total_points=None,
        current_best_is_exact=None,
        current_best_is_terminal=None,
        current_best_source=None,
    )


def default_morpion_frontier_status() -> MorpionBootstrapFrontierStatus:
    """Return the default empty frontier/debug status for the experiment."""
    return MorpionBootstrapFrontierStatus(
        variant=MORPION_BOOTSTRAP_VARIANT,
        initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN,
        initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
        current_best_moves_since_start=None,
        current_best_total_points=None,
        current_best_is_exact=None,
        current_best_is_terminal=None,
        current_best_source=None,
    )


def current_record_score(
    status: MorpionBootstrapRecordStatus,
) -> int | None:
    """Return the canonical Morpion certified-record score."""
    return status.current_best_moves_since_start


def current_frontier_score(
    status: MorpionBootstrapFrontierStatus,
) -> int | None:
    """Return the canonical Morpion frontier/debug score."""
    return status.current_best_moves_since_start


def morpion_score_from_snapshot_depth(
    depth: int,
    root_moves_offset: int = 0,
) -> int:
    """Return Morpion moves played from cheap snapshot depth metadata."""
    return root_moves_offset + depth


def carried_forward_morpion_record_status(
    previous: MorpionBootstrapRecordStatus | None,
) -> MorpionBootstrapRecordStatus:
    """Return the certified record status carried forward across cycles."""
    if previous is None:
        return default_morpion_record_status()
    return _normalized_record_status(previous)


def carried_forward_morpion_frontier_status(
    previous: MorpionBootstrapFrontierStatus | None,
) -> MorpionBootstrapFrontierStatus:
    """Return the frontier/debug status carried forward across cycles."""
    if previous is None:
        return default_morpion_frontier_status()
    return _normalized_frontier_status(previous)


def extract_certified_record_candidates_from_training_tree_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    variant: str = MORPION_BOOTSTRAP_VARIANT,
    initial_pattern: str = MORPION_BOOTSTRAP_INITIAL_PATTERN,
    initial_point_count: int = MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
) -> tuple[MorpionCertifiedRecordCandidate, ...]:
    """Extract every certified Morpion candidate from one training snapshot."""
    scan_started_at = time.perf_counter()
    LOGGER.info("[record] scan_start nodes=%s", len(snapshot.nodes))
    candidates: list[MorpionCertifiedRecordCandidate] = []
    try:
        for node in snapshot.nodes:
            if (
                node.state_ref_payload is None
                or not node.is_terminal
                or not node.is_exact
            ):
                continue
            try:
                normalized_payload, moves_since_start = _decoded_payload_and_moves(
                    node=node,
                    variant=variant,
                )
            except InvalidMorpionStateRefPayloadError:
                continue
            candidate = MorpionCertifiedRecordCandidate(
                node_id=node.node_id,
                variant=variant,
                initial_pattern=initial_pattern,
                initial_point_count=initial_point_count,
                moves_since_start=moves_since_start,
                total_points=initial_point_count + moves_since_start,
                state_ref_payload=normalized_payload,
            )
            LOGGER.info(
                "[record] certified_candidate_found total_points=%s node_id=%s",
                candidate.total_points,
                candidate.node_id,
            )
            candidates.append(candidate)
    finally:
        LOGGER.info(
            "[record] scan_done elapsed=%.3fs num_candidates=%s",
            time.perf_counter() - scan_started_at,
            len(candidates),
        )
    return tuple(candidates)


def extract_morpion_record_status_from_training_tree_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    variant: str = MORPION_BOOTSTRAP_VARIANT,
    initial_pattern: str = MORPION_BOOTSTRAP_INITIAL_PATTERN,
    initial_point_count: int = MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
) -> MorpionBootstrapRecordStatus:
    """Extract the best strict certified Morpion record from one training snapshot."""
    candidates = extract_certified_record_candidates_from_training_tree_snapshot(
        snapshot,
        variant=variant,
        initial_pattern=initial_pattern,
        initial_point_count=initial_point_count,
    )
    best_candidate = _best_certified_candidate(candidates)
    if best_candidate is None:
        return default_morpion_record_status()
    return _record_status_from_candidate(best_candidate)


def select_best_certified_record_candidate_from_training_tree_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    variant: str = MORPION_BOOTSTRAP_VARIANT,
    initial_pattern: str = MORPION_BOOTSTRAP_INITIAL_PATTERN,
    initial_point_count: int = MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
) -> MorpionCertifiedRecordCandidate | None:
    """Return the strongest strict certified Morpion candidate from one snapshot."""
    candidates = extract_certified_record_candidates_from_training_tree_snapshot(
        snapshot,
        variant=variant,
        initial_pattern=initial_pattern,
        initial_point_count=initial_point_count,
    )
    return _best_certified_candidate(candidates)


def extract_morpion_frontier_status_from_training_tree_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    variant: str = MORPION_BOOTSTRAP_VARIANT,
    initial_pattern: str = MORPION_BOOTSTRAP_INITIAL_PATTERN,
    initial_point_count: int = MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    root_moves_offset: int = 0,
) -> MorpionBootstrapFrontierStatus:
    """Extract the best permissive frontier/debug Morpion status from one snapshot."""
    candidates = extract_top_morpion_frontier_nodes_from_training_tree_snapshot(
        snapshot,
        limit=1,
        initial_point_count=initial_point_count,
        root_moves_offset=root_moves_offset,
    )
    return _frontier_status_from_candidates(
        candidates,
        variant=variant,
        initial_pattern=initial_pattern,
        initial_point_count=initial_point_count,
    )


def _frontier_status_from_candidates(
    candidates: tuple[MorpionFrontierNodeCandidate, ...],
    *,
    variant: str,
    initial_pattern: str,
    initial_point_count: int,
) -> MorpionBootstrapFrontierStatus:
    """Return the compact status representation for ordered frontier candidates."""
    if not candidates:
        return MorpionBootstrapFrontierStatus(
            variant=variant,
            initial_pattern=initial_pattern,
            initial_point_count=initial_point_count,
            current_best_moves_since_start=None,
            current_best_total_points=None,
            current_best_is_exact=None,
            current_best_is_terminal=None,
            current_best_source=None,
        )
    best_candidate = candidates[0]
    return MorpionBootstrapFrontierStatus(
        variant=variant,
        initial_pattern=initial_pattern,
        initial_point_count=initial_point_count,
        current_best_moves_since_start=best_candidate.moves_since_start,
        current_best_total_points=best_candidate.total_points,
        current_best_is_exact=best_candidate.is_exact,
        current_best_is_terminal=best_candidate.is_terminal,
        current_best_source=best_candidate.source,
    )


def extract_top_morpion_frontier_nodes_from_training_tree_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    limit: int = 100,
    initial_point_count: int = MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    root_moves_offset: int = 0,
) -> tuple[MorpionFrontierNodeCandidate, ...]:
    """Return top metadata-only frontier candidates without decoding node states."""
    if limit <= 0:
        return ()

    terminal_candidates: list[MorpionFrontierNodeCandidate] = []
    fallback_candidates: list[MorpionFrontierNodeCandidate] = []
    for node in snapshot.nodes:
        moves_since_start = morpion_score_from_snapshot_depth(
            node.depth,
            root_moves_offset=root_moves_offset,
        )
        candidate = MorpionFrontierNodeCandidate(
            node_id=node.node_id,
            moves_since_start=moves_since_start,
            total_points=initial_point_count + moves_since_start,
            is_terminal=node.is_terminal,
            is_exact=node.is_exact,
            source=_snapshot_source_for_node(node),
        )
        if node.is_terminal:
            terminal_candidates.append(candidate)
        else:
            fallback_candidates.append(candidate)

    selected = sorted(terminal_candidates, key=_frontier_candidate_sort_key)[:limit]
    if len(selected) < limit:
        selected.extend(
            sorted(fallback_candidates, key=_frontier_candidate_sort_key)[
                : limit - len(selected)
            ]
        )
    return tuple(selected)


def resolve_record_status_for_cycle(
    *,
    snapshot: TrainingTreeSnapshot | None,
    previous_record_status: MorpionBootstrapRecordStatus | None,
) -> MorpionBootstrapRecordStatus:
    """Resolve the strict certified record status that should be written for one cycle."""
    previous_status = carried_forward_morpion_record_status(previous_record_status)
    if snapshot is None:
        return previous_status

    snapshot_status = extract_morpion_record_status_from_training_tree_snapshot(
        snapshot
    )
    if snapshot_status.current_best_total_points is None:
        LOGGER.info("[record] no_certified_candidate_in_snapshot")
        return previous_status
    resolved_status = _max_record_status(previous_status, snapshot_status)
    if resolved_status != previous_status:
        LOGGER.info(
            "[record] certified_record_updated old=%s new=%s",
            previous_status.current_best_total_points,
            resolved_status.current_best_total_points,
        )
    return resolved_status


def resolve_frontier_status_for_cycle(
    *,
    snapshot: TrainingTreeSnapshot | None,
    previous_frontier_status: MorpionBootstrapFrontierStatus | None,
) -> MorpionBootstrapFrontierStatus:
    """Resolve the permissive frontier/debug status that should be written for one cycle."""
    previous_status = carried_forward_morpion_frontier_status(previous_frontier_status)
    if snapshot is None:
        return previous_status
    snapshot_status = extract_morpion_frontier_status_from_training_tree_snapshot(
        snapshot
    )
    return _max_frontier_status(previous_status, snapshot_status)


def resolve_frontier_status_for_cycle_with_metadata(
    *,
    snapshot: TrainingTreeSnapshot | None,
    previous_frontier_status: MorpionBootstrapFrontierStatus | None,
) -> MorpionFrontierResolution:
    """Resolve frontier status and report actual top-candidate count."""
    previous_status = carried_forward_morpion_frontier_status(previous_frontier_status)
    if snapshot is None:
        return MorpionFrontierResolution(status=previous_status, candidate_count=0)

    candidates = extract_top_morpion_frontier_nodes_from_training_tree_snapshot(
        snapshot
    )
    snapshot_status = _frontier_status_from_candidates(
        candidates,
        variant=MORPION_BOOTSTRAP_VARIANT,
        initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN,
        initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    )
    return MorpionFrontierResolution(
        status=_max_frontier_status(previous_status, snapshot_status),
        candidate_count=len(candidates),
    )


def persist_certified_leaderboard_candidates(
    *,
    snapshot: TrainingTreeSnapshot,
    run_work_dir: str | Path,
    generation: int,
    cycle_index: int,
    timestamp_utc: str,
    leaderboard_path: str | Path | None = None,
) -> None:
    """Update the persistent all-time certified leaderboard from one snapshot."""
    candidates = extract_certified_record_candidates_from_training_tree_snapshot(
        snapshot
    )
    if not candidates:
        return
    resolved_path = (
        Path.home() / "morpion_runs" / "morpion_leaderboard.jsonl"
        if leaderboard_path is None
        else Path(leaderboard_path)
    )
    entries = _load_leaderboard_entries(resolved_path)

    for candidate in candidates:
        fingerprint = fingerprint_morpion_state_payload(
            variant=candidate.variant,
            state_ref_payload=candidate.state_ref_payload,
        )
        LOGGER.info(
            "[leaderboard] candidate variant=%s total_points=%s fingerprint=%s",
            candidate.variant,
            candidate.total_points,
            fingerprint,
        )
        existing_index = next(
            (
                index
                for index, entry in enumerate(entries)
                if entry.state_fingerprint == fingerprint
            ),
            None,
        )
        new_entry = MorpionLeaderboardEntry(
            variant=candidate.variant,
            total_points=candidate.total_points,
            moves_since_start=candidate.moves_since_start,
            is_terminal=True,
            is_exact=True,
            source="certified_terminal_leaf",
            state_fingerprint=fingerprint,
            state_ref_payload=dict(candidate.state_ref_payload),
            run_work_dir=str(Path(run_work_dir)),
            generation=generation,
            cycle_index=cycle_index,
            timestamp_utc=timestamp_utc,
        )
        if existing_index is not None:
            existing_entry = entries[existing_index]
            if _leaderboard_entry_sort_key(
                existing_entry
            ) <= _leaderboard_entry_sort_key(new_entry):
                LOGGER.info(
                    "[leaderboard] skipped_duplicate fingerprint=%s", fingerprint
                )
                continue
            entries[existing_index] = new_entry
        else:
            entries.append(new_entry)

        entries = _top_leaderboard_entries(entries)
        if any(entry.state_fingerprint == fingerprint for entry in entries):
            LOGGER.info("[leaderboard] inserted fingerprint=%s", fingerprint)
        else:
            LOGGER.info("[leaderboard] skipped_not_top_100 fingerprint=%s", fingerprint)
    _save_leaderboard_entries(resolved_path, entries)


def fingerprint_morpion_state_payload(
    *,
    variant: str,
    state_ref_payload: dict[str, object],
) -> str:
    """Return a deterministic sha256 fingerprint for one Morpion state payload."""
    canonical_payload = json.dumps(
        {
            "variant": variant,
            "state_ref_payload": state_ref_payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical_payload).hexdigest()}"


def _decoded_payload_and_moves(
    *,
    node: TrainingNodeSnapshot,
    variant: str,
) -> tuple[dict[str, object], int]:
    """Return normalized payload and move count for one valid Morpion snapshot node."""
    if node.state_ref_payload is None:
        raise InvalidMorpionStateRefPayloadError.payload_must_be_mapping()
    normalized_payload = dict(node.state_ref_payload)
    decoded_state = decode_morpion_state_ref_payload(normalized_payload)
    if decoded_state.variant.value != variant:
        raise InvalidMorpionStateRefPayloadError.payload_not_decodable()
    return normalized_payload, int(decoded_state.moves)


def _best_certified_candidate(
    candidates: tuple[MorpionCertifiedRecordCandidate, ...],
) -> MorpionCertifiedRecordCandidate | None:
    """Return the strongest certified candidate from one snapshot."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (candidate.total_points, candidate.moves_since_start),
    )


def _record_status_from_candidate(
    candidate: MorpionCertifiedRecordCandidate,
) -> MorpionBootstrapRecordStatus:
    """Return one strict certified record status from a certified candidate."""
    return MorpionBootstrapRecordStatus(
        variant=candidate.variant,
        initial_pattern=candidate.initial_pattern,
        initial_point_count=candidate.initial_point_count,
        current_best_moves_since_start=candidate.moves_since_start,
        current_best_total_points=candidate.total_points,
        current_best_is_exact=True,
        current_best_is_terminal=True,
        current_best_source="certified_terminal_leaf",
    )


def _max_record_status(
    previous: MorpionBootstrapRecordStatus,
    current: MorpionBootstrapRecordStatus,
) -> MorpionBootstrapRecordStatus:
    """Return the stronger strict certified record between two statuses."""
    return (
        current
        if _record_status_rank(current) > _record_status_rank(previous)
        else previous
    )


def _max_frontier_status(
    previous: MorpionBootstrapFrontierStatus,
    current: MorpionBootstrapFrontierStatus,
) -> MorpionBootstrapFrontierStatus:
    """Return the stronger frontier/debug status between two statuses."""
    return (
        current
        if _frontier_status_rank(current) > _frontier_status_rank(previous)
        else previous
    )


def _record_status_rank(status: MorpionBootstrapRecordStatus) -> tuple[int, int, int]:
    """Return a stable rank tuple for one certified record status."""
    return (
        -1
        if status.current_best_total_points is None
        else status.current_best_total_points,
        -1
        if status.current_best_moves_since_start is None
        else status.current_best_moves_since_start,
        1 if status.current_best_is_terminal else 0,
    )


def _frontier_status_rank(
    status: MorpionBootstrapFrontierStatus,
) -> tuple[int, int, int, int]:
    """Return a stable rank tuple for one frontier/debug status."""
    return (
        -1
        if status.current_best_total_points is None
        else status.current_best_total_points,
        -1
        if status.current_best_moves_since_start is None
        else status.current_best_moves_since_start,
        1 if status.current_best_is_terminal else 0,
        1 if status.current_best_is_exact else 0,
    )


def _frontier_candidate_sort_key(
    candidate: MorpionFrontierNodeCandidate,
) -> tuple[int, int, int, str]:
    """Return deterministic strongest-first metadata frontier ordering."""
    return (
        -candidate.moves_since_start,
        -int(candidate.is_terminal),
        -int(candidate.is_exact),
        candidate.node_id,
    )


def _normalized_record_status(
    status: MorpionBootstrapRecordStatus,
) -> MorpionBootstrapRecordStatus:
    """Fill missing experiment identity fields without changing known certified data."""
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
        current_best_is_terminal=status.current_best_is_terminal,
        current_best_source=status.current_best_source,
    )


def _normalized_frontier_status(
    status: MorpionBootstrapFrontierStatus,
) -> MorpionBootstrapFrontierStatus:
    """Fill missing experiment identity fields without changing known frontier data."""
    return MorpionBootstrapFrontierStatus(
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
        current_best_is_terminal=status.current_best_is_terminal,
        current_best_source=status.current_best_source,
    )


def _snapshot_source_for_node(node: TrainingNodeSnapshot) -> str:
    """Return the canonical source label for one snapshot-derived frontier node."""
    if node.is_terminal and node.is_exact:
        return "certified_terminal_leaf"
    if node.is_terminal:
        return "snapshot_terminal_node"
    if node.is_exact:
        return "snapshot_exact_node"
    return "snapshot_nonterminal_node"


def _load_leaderboard_entries(path: Path) -> list[MorpionLeaderboardEntry]:
    """Load persisted leaderboard entries when the file exists."""
    if not path.exists():
        return []
    entries: list[MorpionLeaderboardEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        entries.append(
            MorpionLeaderboardEntry(
                variant=str(payload["variant"]),
                total_points=int(payload["total_points"]),
                moves_since_start=int(payload["moves_since_start"]),
                is_terminal=bool(payload["is_terminal"]),
                is_exact=bool(payload["is_exact"]),
                source=str(payload["source"]),
                state_fingerprint=str(payload["state_fingerprint"]),
                state_ref_payload=dict(payload["state_ref_payload"]),
                run_work_dir=str(payload["run_work_dir"]),
                generation=int(payload["generation"]),
                cycle_index=int(payload["cycle_index"]),
                timestamp_utc=str(payload["timestamp_utc"]),
            )
        )
    return entries


def _save_leaderboard_entries(
    path: Path,
    entries: list[MorpionLeaderboardEntry],
) -> None:
    """Persist the compact certified leaderboard as append-friendly JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_leaderboard_entry_to_dict(entry), sort_keys=True) + "\n"
            for entry in entries
        ),
        encoding="utf-8",
    )


def _leaderboard_entry_to_dict(entry: MorpionLeaderboardEntry) -> dict[str, object]:
    """Serialize one leaderboard entry to JSON-friendly data."""
    return {
        "variant": entry.variant,
        "total_points": entry.total_points,
        "moves_since_start": entry.moves_since_start,
        "is_terminal": entry.is_terminal,
        "is_exact": entry.is_exact,
        "source": entry.source,
        "state_fingerprint": entry.state_fingerprint,
        "state_ref_payload": dict(entry.state_ref_payload),
        "run_work_dir": entry.run_work_dir,
        "generation": entry.generation,
        "cycle_index": entry.cycle_index,
        "timestamp_utc": entry.timestamp_utc,
    }


def _top_leaderboard_entries(
    entries: list[MorpionLeaderboardEntry],
) -> list[MorpionLeaderboardEntry]:
    """Return the top distinct certified entries per variant."""
    variants = sorted({entry.variant for entry in entries})
    kept: list[MorpionLeaderboardEntry] = []
    for variant in variants:
        variant_entries = sorted(
            (entry for entry in entries if entry.variant == variant),
            key=_leaderboard_entry_sort_key,
        )
        kept.extend(variant_entries[:MORPION_LEADERBOARD_LIMIT_PER_VARIANT])
    return kept


def _leaderboard_entry_sort_key(entry: MorpionLeaderboardEntry) -> tuple[int, str]:
    """Return the canonical sort key for leaderboard ranking."""
    return (-entry.total_points, entry.timestamp_utc)


__all__ = [
    "MORPION_BOOTSTRAP_GAME",
    "MORPION_BOOTSTRAP_INITIAL_PATTERN",
    "MORPION_BOOTSTRAP_INITIAL_POINT_COUNT",
    "MORPION_BOOTSTRAP_VARIANT",
    "MORPION_LEADERBOARD_LIMIT_PER_VARIANT",
    "MorpionBootstrapFrontierStatus",
    "MorpionBootstrapRecordStatus",
    "MorpionCertifiedRecordCandidate",
    "MorpionFrontierNodeCandidate",
    "MorpionFrontierResolution",
    "MorpionLeaderboardEntry",
    "carried_forward_morpion_frontier_status",
    "carried_forward_morpion_record_status",
    "current_frontier_score",
    "current_record_score",
    "default_morpion_frontier_status",
    "default_morpion_record_status",
    "extract_certified_record_candidates_from_training_tree_snapshot",
    "extract_morpion_frontier_status_from_training_tree_snapshot",
    "extract_morpion_record_status_from_training_tree_snapshot",
    "extract_top_morpion_frontier_nodes_from_training_tree_snapshot",
    "fingerprint_morpion_state_payload",
    "morpion_bootstrap_experiment_metadata",
    "morpion_score_from_snapshot_depth",
    "persist_certified_leaderboard_candidates",
    "resolve_frontier_status_for_cycle",
    "resolve_frontier_status_for_cycle_with_metadata",
    "resolve_record_status_for_cycle",
    "select_best_certified_record_candidate_from_training_tree_snapshot",
]
