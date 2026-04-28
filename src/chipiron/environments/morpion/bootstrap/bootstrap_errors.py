"""Custom exceptions used by the Morpion bootstrap workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class EmptyMorpionEvaluatorsConfigError(ValueError):
    """Raised when the bootstrap loop is configured with zero evaluators."""

    def __init__(self) -> None:
        """Initialize the empty-evaluators-config error."""
        super().__init__(
            "Morpion bootstrap evaluators_config must contain at least one evaluator."
        )


class InvalidBootstrapArtifactPathError(ValueError):
    """Raised when a persisted bootstrap artifact path escapes the work directory."""

    def __init__(self, artifact_path: Path, work_dir: Path) -> None:
        """Initialize the invalid-artifact-path error."""
        super().__init__(
            f"Bootstrap artifact path {artifact_path} must be inside work_dir "
            f"{work_dir} to be persisted relatively."
        )


class InvalidGenerationRetentionCountError(ValueError):
    """Raised when retention configuration requests fewer than one artifact."""

    def __init__(self, keep_latest: int) -> None:
        """Initialize the invalid-retention-count error."""
        super().__init__(
            f"Retention keep_latest must be at least 1, got {keep_latest}."
        )


class InconsistentMorpionEvaluatorSpecNameError(ValueError):
    """Raised when one evaluator spec name does not match its config key."""

    def __init__(self, key: str, spec_name: str) -> None:
        """Initialize the mismatched-evaluator-name error."""
        super().__init__(
            "Morpion bootstrap evaluator config keys must match spec names, got "
            f"key={key!r} and spec.name={spec_name!r}."
        )


class NoSelectableMorpionEvaluatorError(ValueError):
    """Raised when no evaluator can be selected as the active search model."""

    def __init__(self) -> None:
        """Initialize the missing-selectable-evaluator error."""
        super().__init__(
            "Morpion bootstrap could not select an active evaluator because no "
            "trained evaluator reported a finite final_loss."
        )


class UnknownActiveMorpionEvaluatorError(ValueError):
    """Raised when persisted active evaluator state does not match saved bundles."""

    def __init__(self, evaluator_name: str) -> None:
        """Initialize the missing-active-evaluator error."""
        super().__init__(
            "Morpion bootstrap run state refers to active evaluator "
            f"{evaluator_name!r}, but no saved model bundle path exists for it."
        )


class IncompatibleMorpionResumeArtifactError(ValueError):
    """Raised when bootstrap resume selects an artifact that is not a checkpoint."""

    def __init__(
        self,
        *,
        source: str,
        artifact_path: Path,
        reason: str,
    ) -> None:
        """Initialize the incompatible-resume-artifact error."""
        super().__init__(
            "Morpion bootstrap resume selected an incompatible artifact from "
            f"{source}: {artifact_path}. Runtime resume requires a search checkpoint "
            "from `search_checkpoints/...`, while tree exports in "
            "`tree_exports/...` are only for dataset extraction and analysis. "
            f"Details: {reason}"
        )


class MissingActiveMorpionEvaluatorError(ValueError):
    """Raised when persisted multi-evaluator state has no selected active evaluator."""

    def __init__(self) -> None:
        """Initialize the missing-active-evaluator error."""
        super().__init__(
            "Morpion bootstrap run state contains multiple saved evaluator bundles "
            "but no active_evaluator_name, so resume is ambiguous."
        )


class UnknownForcedMorpionEvaluatorError(ValueError):
    """Raised when one control file forces an evaluator name that is unavailable."""

    def __init__(self, evaluator_name: str) -> None:
        """Initialize the forced-evaluator validation error."""
        super().__init__(
            f"Morpion bootstrap control refers to unknown evaluator {evaluator_name!r}."
        )


class UnsupportedMorpionRuntimeReconfigurationError(ValueError):
    """Raised when a requested runtime change is outside the supported safe subset."""

    def __init__(
        self,
        *,
        previous_tree_branch_limit: int,
        requested_tree_branch_limit: int,
    ) -> None:
        """Initialize the unsupported runtime reconfiguration error."""
        super().__init__(
            "Morpion bootstrap supports only non-increasing tree_branch_limit "
            "changes on an existing persisted tree. Requested "
            f"{requested_tree_branch_limit} after {previous_tree_branch_limit}."
        )


class MissingForcedMorpionEvaluatorBundleError(ValueError):
    """Raised when a forced evaluator has no saved bundle at restore time."""

    def __init__(self, evaluator_name: str) -> None:
        """Initialize the missing-forced-bundle error."""
        super().__init__(
            "Morpion bootstrap control forces evaluator "
            f"{evaluator_name!r}, but no saved model bundle path exists for it."
        )


class ConflictingMorpionEvaluatorConfigurationError(ValueError):
    """Raised when bootstrap args specify both explicit config and a family preset."""

    def __init__(self) -> None:
        """Initialize the ambiguous-evaluator-configuration error."""
        super().__init__(
            "Morpion bootstrap args cannot specify both `evaluators_config` and "
            "`evaluator_family_preset`; choose one configuration path."
        )


class MissingSavedBootstrapArtifactError(FileNotFoundError):
    """Raised when a save hook returns without producing its expected artifact."""

    def __init__(self, *, action: str, artifact_path: Path) -> None:
        """Initialize the missing-saved-artifact error."""
        super().__init__(
            f"Morpion bootstrap {action} did not create expected artifact: {artifact_path}"
        )


class UnexpectedBootstrapInvariantError(RuntimeError):
    """Raised when an internal bootstrap helper returns an impossible result."""


class MissingBootstrapRecordStatusError(UnexpectedBootstrapInvariantError):
    """Raised when record-status resolution unexpectedly returns ``None``."""

    def __init__(self) -> None:
        """Initialize the missing-record-status error."""
        super().__init__(
            "Bootstrap invariant violation: record status resolution returned None "
            "after resolve_record_status_for_cycle()."
        )


class MissingBootstrapFrontierStatusError(UnexpectedBootstrapInvariantError):
    """Raised when frontier-status resolution unexpectedly returns ``None``."""

    def __init__(self) -> None:
        """Initialize the missing-frontier-status error."""
        super().__init__(
            "Bootstrap invariant violation: frontier status resolution returned "
            "None after resolve_frontier_status_for_cycle_with_metadata()."
        )


class MissingBootstrapDatasetRowsError(UnexpectedBootstrapInvariantError):
    """Raised when dataset extraction unexpectedly returns ``None``."""

    def __init__(self) -> None:
        """Initialize the missing-dataset-rows error."""
        super().__init__(
            "Bootstrap invariant violation: dataset extraction returned None "
            "after training_tree_snapshot_to_morpion_supervised_rows()."
        )


class MissingBootstrapSelectedEvaluatorError(UnexpectedBootstrapInvariantError):
    """Raised when evaluator selection unexpectedly returns ``None``."""

    def __init__(self) -> None:
        """Initialize the missing-selected-evaluator error."""
        super().__init__(
            "Bootstrap invariant violation: evaluator selection returned None "
            "after _select_active_evaluator_name()."
        )


__all__ = [
    "ConflictingMorpionEvaluatorConfigurationError",
    "EmptyMorpionEvaluatorsConfigError",
    "IncompatibleMorpionResumeArtifactError",
    "InconsistentMorpionEvaluatorSpecNameError",
    "InvalidBootstrapArtifactPathError",
    "InvalidGenerationRetentionCountError",
    "MissingActiveMorpionEvaluatorError",
    "MissingBootstrapDatasetRowsError",
    "MissingBootstrapFrontierStatusError",
    "MissingBootstrapRecordStatusError",
    "MissingBootstrapSelectedEvaluatorError",
    "MissingForcedMorpionEvaluatorBundleError",
    "MissingSavedBootstrapArtifactError",
    "NoSelectableMorpionEvaluatorError",
    "UnexpectedBootstrapInvariantError",
    "UnknownActiveMorpionEvaluatorError",
    "UnknownForcedMorpionEvaluatorError",
    "UnsupportedMorpionRuntimeReconfigurationError",
]
