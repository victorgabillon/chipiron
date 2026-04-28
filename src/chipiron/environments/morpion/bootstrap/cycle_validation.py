"""Validation helpers shared by Morpion bootstrap workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .bootstrap_errors import (
    UnknownForcedMorpionEvaluatorError,
    UnsupportedMorpionRuntimeReconfigurationError,
)
from .control import (
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    MorpionBootstrapEffectiveRuntimeConfig,
    effective_runtime_config_from_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .bootstrap_args import MorpionBootstrapArgs
    from .config import MorpionBootstrapConfig
    from .evaluator_config import MorpionEvaluatorSpec
    from .pipeline_config import MorpionEvaluatorUpdatePolicy


_INVALID_DATASET_FAMILY_BLEND_ERROR = (
    "dataset_family_prediction_blend must be between 0 and 1."
)


def _unknown_pipeline_mode_error(pipeline_mode: object) -> ValueError:
    return ValueError(f"Unknown Morpion pipeline mode: {pipeline_mode!r}")


def _unknown_evaluator_update_policy_error(policy: object) -> ValueError:
    return ValueError(f"Unknown Morpion evaluator update policy: {policy!r}")


def validate_dataset_family_target_args(args: MorpionBootstrapArgs) -> None:
    """Validate dataset-family target blending arguments."""
    if not 0.0 <= args.dataset_family_prediction_blend <= 1.0:
        raise ValueError(_INVALID_DATASET_FAMILY_BLEND_ERROR)


def validate_pipeline_mode(args: MorpionBootstrapArgs) -> None:
    """Validate the configured bootstrap pipeline mode."""
    if args.pipeline_mode == "single_process":
        return
    if args.pipeline_mode == "artifact_pipeline":
        return
    raise _unknown_pipeline_mode_error(args.pipeline_mode)


def require_single_process_mode(args: MorpionBootstrapArgs) -> None:
    """Require explicit single-process mode for the canonical loop entrypoint."""
    if args.pipeline_mode != "single_process":
        raise NotImplementedError(
            "run_morpion_bootstrap_loop only supports pipeline_mode='single_process'. "
            "Use the dedicated artifact-pipeline stage entrypoints instead."
        )


def reevaluate_tree_for_policy(policy: MorpionEvaluatorUpdatePolicy) -> bool:
    """Resolve whether the runner should reevaluate existing tree nodes."""
    if policy == "future_only":
        return False
    if policy == "reevaluate_all":
        return True
    if policy == "reevaluate_frontier":
        raise NotImplementedError(
            "Morpion evaluator_update_policy='reevaluate_frontier' is reserved "
            "for future partial tree reevaluation."
        )
    raise _unknown_evaluator_update_policy_error(policy)


def validate_forced_evaluator(
    *,
    force_evaluator: str | None,
    evaluator_names: Mapping[str, MorpionEvaluatorSpec],
) -> None:
    """Validate one optional forced evaluator against the configured set."""
    if force_evaluator is None:
        return
    if force_evaluator not in evaluator_names:
        raise UnknownForcedMorpionEvaluatorError(force_evaluator)


def previous_effective_runtime_config(
    metadata: Mapping[str, object],
    *,
    resolved_bootstrap_config: MorpionBootstrapConfig,
) -> MorpionBootstrapEffectiveRuntimeConfig | None:
    """Return the last applied runtime config, falling back for legacy metadata."""
    persisted_runtime = effective_runtime_config_from_metadata(
        metadata.get(BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY)
    )
    if persisted_runtime is not None:
        return persisted_runtime
    runtime_checkpoint_path = metadata.get("runtime_checkpoint_path")
    if runtime_checkpoint_path is None:
        return None
    return MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=resolved_bootstrap_config.runtime.tree_branch_limit,
    )


def validate_runtime_reconfiguration(
    *,
    previous_effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> None:
    """Validate that the requested runtime change stays within the supported subset."""
    if previous_effective_runtime_config is None:
        return
    if (
        effective_runtime_config.tree_branch_limit
        > previous_effective_runtime_config.tree_branch_limit
    ):
        raise UnsupportedMorpionRuntimeReconfigurationError(
            previous_tree_branch_limit=(
                previous_effective_runtime_config.tree_branch_limit
            ),
            requested_tree_branch_limit=effective_runtime_config.tree_branch_limit,
        )


__all__ = [
    "previous_effective_runtime_config",
    "reevaluate_tree_for_policy",
    "require_single_process_mode",
    "validate_dataset_family_target_args",
    "validate_forced_evaluator",
    "validate_pipeline_mode",
    "validate_runtime_reconfiguration",
]
