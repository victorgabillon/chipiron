"""Recover stale Morpion artifact-pipeline training claims safely."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineGenerationManifest,
    load_pipeline_manifest,
    save_pipeline_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


DEFAULT_STALE_GRACE_SECONDS = 300


@dataclass(frozen=True, slots=True)
class StaleTrainingState:
    """Classification of one generation's recoverable training state."""

    generation: int
    is_stale: bool
    reason: str
    claim_path: Path | None
    status_path: Path | None
    manifest_path: Path
    claim_expired: bool
    manifest_training_status: str | None
    status_file_status: str | None
    evaluator_results_count: int | None
    dataset_status: str | None = None
    backup_dir: Path | None = None


def inspect_training_state_for_recovery(
    generation_dir: Path,
    *,
    now_utc: datetime,
    stale_grace_seconds: int,
) -> StaleTrainingState:
    """Classify whether one generation has stale training state."""
    manifest_path = generation_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    generation = _generation_from_dir(generation_dir, manifest)
    claim_path = generation_dir / "training_claim.json"
    status_path = generation_dir / "training_status.json"
    claim_payload = _read_json_mapping(claim_path)
    status_payload = _read_json_mapping(status_path)
    claim_exists = claim_path.is_file()
    status_exists = status_path.is_file()
    claim_expired = _claim_expired(
        claim_payload,
        now_utc=now_utc,
        stale_grace_seconds=stale_grace_seconds,
    )
    active_claim = claim_exists and not claim_expired
    evaluator_results_count = _evaluator_results_count(status_payload)
    status_file_status = _str_value(status_payload.get("status"))
    manifest_training_status = None if manifest is None else manifest.training_status
    dataset_status = None if manifest is None else manifest.dataset_status

    base_state = StaleTrainingState(
        generation=generation,
        is_stale=False,
        reason="not_stale",
        claim_path=claim_path if claim_exists else None,
        status_path=status_path if status_exists else None,
        manifest_path=manifest_path,
        claim_expired=claim_expired,
        manifest_training_status=manifest_training_status,
        status_file_status=status_file_status,
        evaluator_results_count=evaluator_results_count,
        dataset_status=dataset_status,
    )
    if manifest is None:
        return dataclass_replace(base_state, reason="missing_manifest")
    if manifest.training_status == "done" or status_file_status == "done":
        return dataclass_replace(base_state, reason="training_done")
    if evaluator_results_count is not None and evaluator_results_count > 0:
        return dataclass_replace(base_state, reason="evaluator_results_present")
    if active_claim:
        return dataclass_replace(base_state, reason="active_training_claim")
    if manifest.dataset_status != "done":
        return dataclass_replace(base_state, reason="dataset_not_done")

    status_training_empty = status_file_status == "training" and (
        evaluator_results_count in (0, None)
    )
    manifest_training = manifest.training_status == "training"
    if status_training_empty and (claim_expired or not claim_exists):
        reason = (
            "expired_training_claim_empty_evaluator_results"
            if claim_expired
            else "missing_training_claim_empty_evaluator_results"
        )
        return dataclass_replace(base_state, is_stale=True, reason=reason)
    if manifest_training and (claim_expired or not claim_exists):
        reason = (
            "expired_training_claim_manifest_training"
            if claim_expired
            else "missing_training_claim_manifest_training"
        )
        return dataclass_replace(base_state, is_stale=True, reason=reason)
    if claim_expired:
        return dataclass_replace(
            base_state,
            is_stale=True,
            reason="expired_training_claim",
        )
    return base_state


def recover_stale_training_state(
    generation_dir: Path,
    stale_state: StaleTrainingState,
    *,
    dry_run: bool = False,
) -> bool:
    """Recover one safely classified stale training state."""
    if not stale_state.is_stale:
        return False
    if dry_run:
        return True

    manifest = load_pipeline_manifest(stale_state.manifest_path)
    if manifest.training_status == "done":
        return False

    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    backup_dir = _unique_backup_dir(
        generation_dir,
        timestamp=timestamp,
    )
    backup_dir.mkdir(parents=True)
    _backup_path(stale_state.manifest_path, backup_dir)

    if stale_state.claim_path is not None and stale_state.claim_path.is_file():
        _backup_path(stale_state.claim_path, backup_dir)
        stale_state.claim_path.unlink()
    if _can_remove_training_status(stale_state):
        assert stale_state.status_path is not None
        _backup_path(stale_state.status_path, backup_dir)
        stale_state.status_path.unlink()

    metadata = dict(manifest.metadata)
    new_training_status = (
        "not_started"
        if manifest.training_status == "training"
        else manifest.training_status
    )
    metadata["auto_recovery"] = {
        "backup_dir": backup_dir.name,
        "new_training_status": new_training_status,
        "old_training_status": manifest.training_status,
        "reason": stale_state.reason,
        "training_status_reset_at_utc": timestamp,
    }
    save_pipeline_manifest(
        dataclass_replace(
            manifest,
            training_status=new_training_status,
            metadata=metadata,
        ),
        stale_state.manifest_path,
    )
    return True


def inspect_training_states_for_work_dir(
    work_dir: Path,
    *,
    generations: tuple[int, ...] | None = None,
    now_utc: datetime | None = None,
    stale_grace_seconds: int = DEFAULT_STALE_GRACE_SECONDS,
) -> tuple[StaleTrainingState, ...]:
    """Inspect training recovery state for one work directory."""
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    now = datetime.now(UTC) if now_utc is None else now_utc
    generation_dirs = _generation_dirs(paths, generations=generations)
    return tuple(
        inspect_training_state_for_recovery(
            generation_dir,
            now_utc=now,
            stale_grace_seconds=stale_grace_seconds,
        )
        for generation_dir in generation_dirs
    )


def _can_remove_training_status(stale_state: StaleTrainingState) -> bool:
    return (
        stale_state.status_path is not None
        and stale_state.status_file_status == "training"
        and stale_state.evaluator_results_count in (0, None)
    )


def _load_manifest(path: Path) -> MorpionPipelineGenerationManifest | None:
    if not path.is_file():
        return None
    try:
        return load_pipeline_manifest(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _generation_from_dir(
    generation_dir: Path,
    manifest: MorpionPipelineGenerationManifest | None,
) -> int:
    if manifest is not None:
        return manifest.generation
    try:
        return int(generation_dir.name.removeprefix("generation_"))
    except ValueError:
        return -1


def _read_json_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return dict(cast("Mapping[str, object]", payload))


def _claim_expired(
    claim_payload: dict[str, object],
    *,
    now_utc: datetime,
    stale_grace_seconds: int,
) -> bool:
    expires_at = _str_value(claim_payload.get("expires_at_utc"))
    if expires_at is None:
        return False
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    expired_before = now_utc.astimezone(UTC) - timedelta(seconds=stale_grace_seconds)
    return parsed.astimezone(UTC) < expired_before


def _evaluator_results_count(payload: dict[str, object]) -> int | None:
    if not payload:
        return None
    evaluator_results = payload.get("evaluator_results")
    if isinstance(evaluator_results, dict):
        return len(evaluator_results)
    return None


def _str_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _unique_backup_dir(generation_dir: Path, *, timestamp: str) -> Path:
    safe_timestamp = timestamp.replace(":", "").replace("-", "").replace(".", "")
    base = generation_dir / f"stale_training_backup_{safe_timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = generation_dir / f"{base.name}_{suffix}"
    return candidate


def _backup_path(path: Path, backup_dir: Path) -> None:
    if path.is_file():
        shutil.copy2(path, backup_dir / path.name)


def _generation_dirs(
    paths: MorpionBootstrapPaths,
    *,
    generations: tuple[int, ...] | None,
) -> tuple[Path, ...]:
    if generations is not None:
        return tuple(
            paths.pipeline_generation_dir_for_generation(gen) for gen in generations
        )
    if not paths.pipeline_dir.is_dir():
        return ()
    return tuple(
        sorted(
            (
                child
                for child in paths.pipeline_dir.iterdir()
                if child.is_dir() and child.name.startswith("generation_")
            ),
            key=lambda path: path.name,
        )
    )


def _render_state(state: StaleTrainingState, *, planned_action: str) -> str:
    return "\n".join((
        f"generation={state.generation}",
        f"stale={str(state.is_stale).lower()}",
        f"reason={state.reason}",
        f"claim_expired={str(state.claim_expired).lower()}",
        f"dataset_status={state.dataset_status}",
        f"manifest_training_status={state.manifest_training_status}",
        f"status_file_status={state.status_file_status}",
        f"evaluator_results={state.evaluator_results_count}",
        f"planned_action={planned_action}",
    ))


def main(argv: list[str] | None = None) -> int:
    """Run the stale training recovery CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--all", action="store_true", dest="scan_all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument(
        "--stale-grace-seconds",
        type=int,
        default=DEFAULT_STALE_GRACE_SECONDS,
    )
    args = parser.parse_args(argv)

    if args.generation is None and not args.scan_all:
        parser.error("expected --generation or --all")
    if args.dry_run and args.recover:
        parser.error("--dry-run and --recover are mutually exclusive")

    generations = None if args.scan_all else (args.generation,)
    assert generations is None or generations[0] is not None
    states = inspect_training_states_for_work_dir(
        args.work_dir,
        generations=cast("tuple[int, ...] | None", generations),
        stale_grace_seconds=args.stale_grace_seconds,
    )
    for state in states:
        planned_action = "reset_training_to_not_started" if state.is_stale else "none"
        print(_render_state(state, planned_action=planned_action))
        if args.recover and state.is_stale:
            recovered = recover_stale_training_state(
                state.manifest_path.parent,
                state,
            )
            print(f"recovered={str(recovered).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
