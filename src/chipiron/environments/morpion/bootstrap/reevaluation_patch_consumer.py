"""Consumption helpers for singleton Morpion reevaluation patches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .pipeline_artifacts import (
    MissingMorpionPipelineArtifactError,
    delete_reevaluation_patch,
    load_reevaluation_patch,
)
from .pipeline_memory import log_pipeline_memory

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import MorpionReevaluationPatch

LOGGER = logging.getLogger(__name__)


def _metric_value(value: object | None) -> str:
    """Render one optional reevaluation metric for structured logs."""
    if value is None:
        return "none"
    return str(value)


def _missing_runner_patch_hook_error() -> NotImplementedError:
    """Build the stable missing-runner patch hook error."""
    return NotImplementedError(
        "Morpion search runner must implement apply_reevaluation_patch() to consume reevaluation patches."
    )


def _invalid_patch_apply_return_type_error() -> TypeError:
    """Build the stable invalid patch-apply return-type error."""
    return TypeError("apply_reevaluation_patch() must return int or None")


def _negative_patch_apply_return_error(applied_count: int) -> ValueError:
    """Build the stable invalid negative patch-apply count error."""
    return ValueError(
        f"apply_reevaluation_patch() must return a non-negative int, got {applied_count}"
    )


@dataclass(frozen=True, slots=True)
class MorpionReevaluationPatchConsumptionResult:
    """Summary of one attempted reevaluation patch consumption."""

    patch_found: bool
    patch_applied: bool
    patch_id: str | None
    num_rows: int
    reason: str | None


def _resolve_applied_count(
    patch: MorpionReevaluationPatch,
    raw_applied_count: object,
) -> int:
    """Normalize the runner hook return into a validated applied-row count."""
    if raw_applied_count is None:
        return len(patch.rows)
    if isinstance(raw_applied_count, bool) or not isinstance(raw_applied_count, int):
        raise _invalid_patch_apply_return_type_error()
    if raw_applied_count < 0:
        raise _negative_patch_apply_return_error(raw_applied_count)
    return raw_applied_count


def apply_pending_reevaluation_patch_to_runner(
    *,
    paths: MorpionBootstrapPaths,
    runner: object,
) -> MorpionReevaluationPatchConsumptionResult:
    """Apply and delete one pending reevaluation patch if it exists."""
    patch_path = paths.pipeline_reevaluation_patch_path
    log_pipeline_memory(
        stage="reevaluation",
        event="start",
        patch_path=patch_path,
    )
    if not patch_path.exists():
        LOGGER.debug("[reevaluation-patch] missing path=%s", str(patch_path))
        log_pipeline_memory(
            stage="reevaluation",
            event="done",
            rows=0,
            reason="missing_patch",
        )
        return MorpionReevaluationPatchConsumptionResult(
            patch_found=False,
            patch_applied=False,
            patch_id=None,
            num_rows=0,
            reason="missing_patch",
        )

    try:
        patch = load_reevaluation_patch(patch_path)
    except MissingMorpionPipelineArtifactError:
        LOGGER.debug("[reevaluation-patch] missing path=%s", str(patch_path))
        log_pipeline_memory(
            stage="reevaluation",
            event="done",
            rows=0,
            reason="missing_patch",
        )
        return MorpionReevaluationPatchConsumptionResult(
            patch_found=False,
            patch_applied=False,
            patch_id=None,
            num_rows=0,
            reason="missing_patch",
        )
    log_pipeline_memory(
        stage="reevaluation",
        generation=patch.tree_generation,
        event="after_patch_load",
        rows=len(patch.rows),
    )

    apply_patch = getattr(runner, "apply_reevaluation_patch", None)
    if not callable(apply_patch):
        raise _missing_runner_patch_hook_error()

    LOGGER.info(
        "[reevaluation-patch] apply_start patch_id=%s rows=%s",
        patch.patch_id,
        len(patch.rows),
    )
    try:
        applied_count = _resolve_applied_count(patch, apply_patch(patch))
    except Exception:
        LOGGER.exception(
            "[reevaluation-patch] apply_fail patch_id=%s rows=%s",
            patch.patch_id,
            len(patch.rows),
        )
        raise

    apply_metrics = getattr(runner, "last_reevaluation_patch_apply_metrics", None)
    if callable(apply_metrics):
        apply_metrics = apply_metrics()
    delete_reevaluation_patch(patch_path)
    LOGGER.info(
        "[reevaluation-patch] apply_done patch_id=%s applied=%s missing=%s recomputed=%s selector_invalidated=%s",
        patch.patch_id,
        applied_count,
        _metric_value(
            cast("dict[str, object]", apply_metrics).get("missing")
            if isinstance(apply_metrics, dict)
            else None
        ),
        _metric_value(
            cast("dict[str, object]", apply_metrics).get("recomputed")
            if isinstance(apply_metrics, dict)
            else None
        ),
        _metric_value(
            cast("dict[str, object]", apply_metrics).get("selector_invalidated")
            if isinstance(apply_metrics, dict)
            else None
        ),
    )
    log_pipeline_memory(
        stage="reevaluation",
        generation=patch.tree_generation,
        event="done",
        rows=applied_count,
    )
    return MorpionReevaluationPatchConsumptionResult(
        patch_found=True,
        patch_applied=True,
        patch_id=patch.patch_id,
        num_rows=applied_count,
        reason=None,
    )


__all__ = [
    "MorpionReevaluationPatchConsumptionResult",
    "apply_pending_reevaluation_patch_to_runner",
]
