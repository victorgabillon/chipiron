"""Temporary stage-claim helpers for the Morpion artifact pipeline."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from .cycle_timing import timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s
from .pipeline_artifacts import (
    InvalidMorpionPipelineArtifactError,
    MissingMorpionPipelineArtifactError,
    MorpionPipelineStageClaim,
    MorpionPipelineStageName,
    delete_pipeline_stage_claim,
    load_pipeline_stage_claim,
    pipeline_stage_claim_to_dict,
    save_pipeline_stage_claim,
)


class PipelineStageAlreadyClaimedError(RuntimeError):
    """Raised when one worker tries to claim a stage that is still active."""


class PipelineStageClaimMismatchError(RuntimeError):
    """Raised when one worker tries to release a claim owned by another worker."""


def _invalid_claim_ttl_error() -> ValueError:
    """Build the stable invalid claim TTL error."""
    return ValueError("claim ttl_seconds must be > 0")


def _already_claimed_error(
    *,
    stage: MorpionPipelineStageName,
    generation: int,
    claim_id: str,
) -> PipelineStageAlreadyClaimedError:
    """Build the stable non-expired-claim error."""
    return PipelineStageAlreadyClaimedError(
        f"pipeline stage already claimed: stage={stage} generation={generation} claim_id={claim_id}"
    )


def _claim_mismatch_error(
    *,
    claim_path: Path,
    expected_claim_id: str,
    actual_claim_id: str,
) -> PipelineStageClaimMismatchError:
    """Build the stable release-mismatch error."""
    return PipelineStageClaimMismatchError(
        "pipeline stage claim mismatch: "
        f"path={claim_path} expected_claim_id={expected_claim_id} actual_claim_id={actual_claim_id}"
    )


def _claim_timestamp_parse_error(
    *,
    field_name: str,
    value: str,
) -> InvalidMorpionPipelineArtifactError:
    """Build the stable invalid-claim-timestamp error."""
    return InvalidMorpionPipelineArtifactError(
        f"Morpion pipeline artifact field `{field_name}` must be an ISO-8601 UTC timestamp; got {value!r}."
    )


def _parse_claim_timestamp_to_unix_s(*, field_name: str, value: str) -> float:
    """Parse one persisted claim timestamp into Unix seconds."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _claim_timestamp_parse_error(field_name=field_name, value=value) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).timestamp()


def _build_stage_claim(
    *,
    generation: int,
    stage: MorpionPipelineStageName,
    now_unix_s: float,
    ttl_seconds: float,
    claim_id: str | None,
    owner: str | None,
    metadata: Mapping[str, object] | None,
) -> MorpionPipelineStageClaim:
    """Build one validated stage-claim record."""
    resolved_claim_id = str(uuid.uuid4()) if claim_id is None else claim_id
    claimed_at_utc = _timestamp_utc_from_unix_s(now_unix_s)
    expires_at_utc = _timestamp_utc_from_unix_s(now_unix_s + ttl_seconds)
    return MorpionPipelineStageClaim(
        generation=generation,
        stage=stage,
        claim_id=resolved_claim_id,
        claimed_at_utc=claimed_at_utc,
        expires_at_utc=expires_at_utc,
        owner=owner,
        metadata={} if metadata is None else dict(metadata),
    )


def _write_claim_json_exclusive(claim: MorpionPipelineStageClaim, claim_path: Path) -> None:
    """Create one claim file using exclusive creation semantics."""
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump(pipeline_stage_claim_to_dict(claim), handle, indent=2, sort_keys=True)
        handle.write("\n")


def pipeline_stage_claim_is_expired(
    claim: MorpionPipelineStageClaim,
    now_unix_s: float | None = None,
) -> bool:
    """Return whether one stage claim has expired."""
    current_time = time.time() if now_unix_s is None else now_unix_s
    expires_at_unix_s = _parse_claim_timestamp_to_unix_s(
        field_name="expires_at_utc",
        value=claim.expires_at_utc,
    )
    return expires_at_unix_s <= current_time


def claim_pipeline_stage(
    *,
    generation: int,
    stage: MorpionPipelineStageName,
    claim_path: Path,
    now_unix_s: float | None = None,
    ttl_seconds: float = 3600.0,
    claim_id: str | None = None,
    owner: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MorpionPipelineStageClaim:
    """Claim one dataset or training stage using a durable claim file."""
    if ttl_seconds <= 0:
        raise _invalid_claim_ttl_error()

    current_time = time.time() if now_unix_s is None else now_unix_s
    next_claim = _build_stage_claim(
        generation=generation,
        stage=stage,
        now_unix_s=current_time,
        ttl_seconds=ttl_seconds,
        claim_id=claim_id,
        owner=owner,
        metadata=metadata,
    )

    for _attempt in range(3):
        try:
            _write_claim_json_exclusive(next_claim, claim_path)
        except FileExistsError:
            pass
        else:
            return next_claim

        try:
            existing_claim = load_pipeline_stage_claim(claim_path)
        except MissingMorpionPipelineArtifactError:
            continue

        if pipeline_stage_claim_is_expired(existing_claim, now_unix_s=current_time):
            save_pipeline_stage_claim(next_claim, claim_path)
            return next_claim

        raise _already_claimed_error(
            stage=stage,
            generation=generation,
            claim_id=existing_claim.claim_id,
        )

    _write_claim_json_exclusive(next_claim, claim_path)
    return next_claim


def load_active_pipeline_stage_claim(
    claim_path: Path,
    now_unix_s: float | None = None,
) -> MorpionPipelineStageClaim | None:
    """Load one non-expired stage claim, returning ``None`` if absent or stale."""
    try:
        claim = load_pipeline_stage_claim(claim_path)
    except MissingMorpionPipelineArtifactError:
        return None
    if pipeline_stage_claim_is_expired(claim, now_unix_s=now_unix_s):
        return None
    return claim


def release_pipeline_stage_claim(*, claim_path: Path, claim_id: str) -> None:
    """Release one stage claim if the caller still owns it."""
    if not claim_path.is_file():
        return
    try:
        claim = load_pipeline_stage_claim(claim_path)
    except MissingMorpionPipelineArtifactError:
        return
    if claim.claim_id != claim_id:
        raise _claim_mismatch_error(
            claim_path=claim_path,
            expected_claim_id=claim_id,
            actual_claim_id=claim.claim_id,
        )
    delete_pipeline_stage_claim(claim_path)


__all__ = [
    "PipelineStageAlreadyClaimedError",
    "PipelineStageClaimMismatchError",
    "claim_pipeline_stage",
    "load_active_pipeline_stage_claim",
    "pipeline_stage_claim_is_expired",
    "release_pipeline_stage_claim",
]
