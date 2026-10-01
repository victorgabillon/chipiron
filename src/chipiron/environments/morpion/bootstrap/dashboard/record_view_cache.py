"""Persistent certified-record state cache for the read-only dashboard."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anemone.checkpoints import checkpoint_payload_to_jsonable
from platformdirs import user_cache_path

from chipiron.environments.morpion.bootstrap.record_status import (
    MorpionBootstrapRecordStatus,
)

LOGGER = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CachedCertifiedRecord:
    """Minimal persisted state needed to reconstruct one certified record board."""

    variant: str
    moves_since_start: int
    total_points: int
    is_exact: bool
    is_terminal: bool
    source: str
    state_ref_payload: object
    node_id: str | None = None
    generation: int | None = None


def load_cached_certified_record(
    *,
    work_dir: str | Path,
    status: MorpionBootstrapRecordStatus,
) -> CachedCertifiedRecord | None:
    """Load one matching dashboard-only record cache entry."""
    signature = _status_signature(status)
    if signature is None:
        return None
    path = _cache_path(Path(work_dir), signature)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        if payload.get("work_dir") != str(Path(work_dir).resolve()):
            return None
        if payload.get("record_signature") != signature:
            return None
        return _cached_record_from_mapping(payload["record"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        LOGGER.warning(
            "[dashboard] certified_record_cache_read_failed path=%s",
            path,
            exc_info=True,
        )
        return None


def save_cached_certified_record(
    *,
    work_dir: str | Path,
    status: MorpionBootstrapRecordStatus,
    record: CachedCertifiedRecord,
) -> None:
    """Persist one dashboard-only record state without modifying the run directory."""
    signature = _status_signature(status)
    if signature is None or not _record_matches_status(record, status):
        return
    path = _cache_path(Path(work_dir), signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "work_dir": str(Path(work_dir).resolve()),
        "record_signature": signature,
        "record": _record_to_mapping(record),
    }
    _write_cache_payload(path, payload)



def load_cached_certified_record_for_snapshot(
    *,
    work_dir: str | Path,
    snapshot_path: Path,
) -> CachedCertifiedRecord | None:
    """Load a dashboard-only record cached for one immutable tree snapshot."""
    signature = _snapshot_signature(snapshot_path)
    if signature is None:
        return None
    path = _cache_path(Path(work_dir), "snapshot:" + signature)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        if payload.get("work_dir") != str(Path(work_dir).resolve()):
            return None
        if payload.get("snapshot_signature") != signature:
            return None
        return _cached_record_from_mapping(payload["record"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        LOGGER.warning(
            "[dashboard] certified_record_snapshot_cache_read_failed path=%s",
            path,
            exc_info=True,
        )
        return None


def save_cached_certified_record_for_snapshot(
    *,
    work_dir: str | Path,
    snapshot_path: Path,
    record: CachedCertifiedRecord,
) -> None:
    """Persist a record state keyed by snapshot identity for legacy runs."""
    signature = _snapshot_signature(snapshot_path)
    if signature is None:
        return
    path = _cache_path(Path(work_dir), "snapshot:" + signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "work_dir": str(Path(work_dir).resolve()),
        "snapshot_signature": signature,
        "record": _record_to_mapping(record),
    }
    _write_cache_payload(path, payload)


def load_matching_leaderboard_record(
    *,
    work_dir: str | Path,
    status: MorpionBootstrapRecordStatus,
    leaderboard_path: str | Path | None = None,
) -> CachedCertifiedRecord | None:
    """Find the current run's certified record in the existing compact leaderboard."""
    if _status_signature(status) is None:
        return None
    path = (
        _default_leaderboard_path()
        if leaderboard_path is None
        else Path(leaderboard_path)
    )
    if not path.is_file():
        return None

    resolved_work_dir = Path(work_dir).resolve()
    matches: list[tuple[int, int, str, CachedCertifiedRecord]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    entry_work_dir = Path(str(payload["run_work_dir"])).resolve()
                    if entry_work_dir != resolved_work_dir:
                        continue
                    record = CachedCertifiedRecord(
                        variant=str(payload["variant"]),
                        moves_since_start=int(payload["moves_since_start"]),
                        total_points=int(payload["total_points"]),
                        is_exact=bool(payload["is_exact"]),
                        is_terminal=bool(payload["is_terminal"]),
                        source=str(payload["source"]),
                        state_ref_payload=payload["state_ref_payload"],
                        generation=int(payload["generation"]),
                    )
                    if not _record_matches_status(record, status):
                        continue
                    matches.append((
                        int(payload["generation"]),
                        int(payload["cycle_index"]),
                        str(payload["timestamp_utc"]),
                        record,
                    ))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    LOGGER.debug(
                        "[dashboard] certified_record_leaderboard_row_skipped path=%s",
                        path,
                        exc_info=True,
                    )
    except OSError:
        LOGGER.warning(
            "[dashboard] certified_record_leaderboard_read_failed path=%s",
            path,
            exc_info=True,
        )
        return None

    if not matches:
        return None
    return max(matches, key=lambda item: item[:3])[3]



def _record_to_mapping(record: CachedCertifiedRecord) -> dict[str, object]:
    """Return the JSON-friendly representation shared by both cache keys."""
    return {
        "variant": record.variant,
        "moves_since_start": record.moves_since_start,
        "total_points": record.total_points,
        "is_exact": record.is_exact,
        "is_terminal": record.is_terminal,
        "source": record.source,
        "state_ref_payload": checkpoint_payload_to_jsonable(record.state_ref_payload),
        "node_id": record.node_id,
        "generation": record.generation,
    }


def _write_cache_payload(path: Path, payload: dict[str, object]) -> None:
    """Atomically write one dashboard-only cache object."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
        _prune_record_cache(path)
    except OSError:
        LOGGER.warning(
            "[dashboard] certified_record_cache_write_failed path=%s",
            path,
            exc_info=True,
        )
        try:
            temporary_path.unlink()
        except OSError:
            pass


def _snapshot_signature(snapshot_path: Path) -> str | None:
    """Return a cheap immutable identity for one training-tree snapshot reference."""
    try:
        resolved = snapshot_path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return json.dumps(
        {
            "path": str(resolved),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _status_signature(status: MorpionBootstrapRecordStatus) -> str | None:
    """Return a stable signature only for a concrete certified record."""
    if (
        status.current_best_total_points is None
        or status.current_best_moves_since_start is None
        or not (
            status.current_best_is_exact is True
            or status.current_best_is_terminal is True
        )
    ):
        return None
    return json.dumps(
        {
            "variant": status.variant,
            "moves_since_start": status.current_best_moves_since_start,
            "total_points": status.current_best_total_points,
            "is_exact": status.current_best_is_exact,
            "is_terminal": status.current_best_is_terminal,
            "source": status.current_best_source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_matches_status(
    record: CachedCertifiedRecord,
    status: MorpionBootstrapRecordStatus,
) -> bool:
    """Require cached state provenance to match the authoritative small status."""
    return (
        status.current_best_total_points == record.total_points
        and status.current_best_moves_since_start == record.moves_since_start
        and (status.variant is None or status.variant == record.variant)
        and (
            status.current_best_source is None
            or status.current_best_source == record.source
        )
        and (
            status.current_best_is_exact is None
            or status.current_best_is_exact == record.is_exact
        )
        and (
            status.current_best_is_terminal is None
            or status.current_best_is_terminal == record.is_terminal
        )
    )


def _cached_record_from_mapping(payload: Any) -> CachedCertifiedRecord:
    """Decode one cached JSON mapping into the typed dashboard record."""
    if not isinstance(payload, dict):
        raise TypeError("cached record must be an object")
    generation = payload.get("generation")
    node_id = payload.get("node_id")
    return CachedCertifiedRecord(
        variant=str(payload["variant"]),
        moves_since_start=int(payload["moves_since_start"]),
        total_points=int(payload["total_points"]),
        is_exact=bool(payload["is_exact"]),
        is_terminal=bool(payload["is_terminal"]),
        source=str(payload["source"]),
        state_ref_payload=payload["state_ref_payload"],
        node_id=None if node_id is None else str(node_id),
        generation=None if generation is None else int(generation),
    )


def _dashboard_cache_root() -> Path:
    """Return the dashboard-only user cache directory."""
    override = os.environ.get("CHIPIRON_DASHBOARD_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return user_cache_path("chipiron") / "bootstrap"


def _cache_path(work_dir: Path, signature: str) -> Path:
    """Return the cache path for one work directory and record signature."""
    work_hash = hashlib.sha256(str(work_dir.resolve()).encode("utf-8")).hexdigest()[:20]
    signature_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return (
        _dashboard_cache_root()
        / "record-view"
        / work_hash
        / f"{signature_hash}.json"
    )


def _prune_record_cache(current_path: Path) -> None:
    """Retain only a small recent set of immutable record states per run."""
    candidates = sorted(
        current_path.parent.glob("*.json"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates[8:]:
        try:
            candidate.unlink()
        except OSError:
            LOGGER.debug(
                "[dashboard] certified_record_cache_prune_failed path=%s",
                candidate,
                exc_info=True,
            )


def _default_leaderboard_path() -> Path:
    """Mirror the existing bootstrap leaderboard location without changing it."""
    return Path.home() / "morpion_runs" / "morpion_leaderboard.jsonl"


__all__ = [
    "CachedCertifiedRecord",
    "load_cached_certified_record",
    "load_cached_certified_record_for_snapshot",
    "load_matching_leaderboard_record",
    "save_cached_certified_record",
    "save_cached_certified_record_for_snapshot",
]
