"""Artifact-driven bootstrap loop for Morpion self-training."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from anemone.training_export import load_training_tree_snapshot
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows,
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)
from .evaluator_family import morpion_evaluators_config_from_preset

from .history import (
    MorpionBootstrapArtifacts,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryPaths,
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
)
from .config import (
    BOOTSTRAP_CONFIG_HASH_METADATA_KEY,
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MorpionBootstrapConfig,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    load_bootstrap_config,
    save_bootstrap_config,
    validate_bootstrap_config_change,
)
from .control import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    apply_control_to_args,
    bootstrap_control_to_dict,
    bootstrap_runtime_control_to_dict,
    effective_runtime_config_from_config_and_control,
    effective_runtime_config_from_metadata,
    effective_runtime_config_sha256,
    effective_runtime_config_to_dict,
    load_bootstrap_control,
)
from .record_status import (
    MorpionBootstrapRecordStatus,
    default_morpion_record_status,
    morpion_bootstrap_experiment_metadata,
    resolve_record_status_for_cycle,
)
from .run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)


def _empty_evaluator_specs() -> dict[str, MorpionEvaluatorSpec]:
    """Return a typed empty evaluator-spec mapping."""
    return {}


RUNTIME_CHECKPOINT_METADATA_KEY = "runtime_checkpoint_path"
TRAINING_SKIPPED_REASON_METADATA_KEY = "training_skipped_reason"
EMPTY_DATASET_TRAINING_SKIPPED_REASON = "empty_dataset"


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


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorSpec:
    """Training spec for one named Morpion evaluator."""

    name: str
    model_type: str
    hidden_sizes: tuple[int, ...] | None
    num_epochs: int
    batch_size: int
    learning_rate: float
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this evaluator."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorsConfig:
    """Deterministic collection of evaluator specs for one bootstrap run."""

    evaluators: dict[str, MorpionEvaluatorSpec] = field(
        default_factory=_empty_evaluator_specs
    )

    def __post_init__(self) -> None:
        """Copy and validate the evaluator mapping eagerly."""
        copied: dict[str, MorpionEvaluatorSpec] = dict(self.evaluators)
        if not copied:
            raise EmptyMorpionEvaluatorsConfigError
        for key, spec in copied.items():
            if key != spec.name:
                raise InconsistentMorpionEvaluatorSpecNameError(key, spec.name)
        object.__setattr__(self, "evaluators", copied)


@dataclass(frozen=True, slots=True)
class MorpionBootstrapArgs:
    """Top-level arguments for the restartable Morpion bootstrap loop."""

    work_dir: str | Path
    max_growth_steps_per_cycle: int = 1000
    save_after_tree_growth_factor: float = 2.0
    save_after_seconds: float = 3600.0
    require_exact_or_terminal: bool = False
    min_depth: int | None = None
    min_visit_count: int | None = None
    max_rows: int | None = None
    use_backed_up_value: bool = True
    tree_branch_limit: int = DEFAULT_MORPION_TREE_BRANCH_LIMIT
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    model_kind: str = "linear"
    hidden_dim: int | None = None
    evaluators_config: MorpionEvaluatorsConfig | None = None
    evaluator_family_preset: str | None = None

    def resolved_evaluators_config(self) -> MorpionEvaluatorsConfig:
        """Resolve the explicit or legacy single-evaluator config."""
        if (
            self.evaluators_config is not None
            and self.evaluator_family_preset is not None
        ):
            raise ConflictingMorpionEvaluatorConfigurationError
        if self.evaluators_config is not None:
            return self.evaluators_config
        if self.evaluator_family_preset is not None:
            return morpion_evaluators_config_from_preset(
                self.evaluator_family_preset
            )
        hidden_sizes = None if self.hidden_dim is None else (self.hidden_dim,)
        default_spec = MorpionEvaluatorSpec(
            name="default",
            model_type=self.model_kind,
            hidden_sizes=hidden_sizes,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        return MorpionEvaluatorsConfig(evaluators={"default": default_spec})


@dataclass(frozen=True, slots=True)
class MorpionBootstrapPaths:
    """Canonical artifact locations for one Morpion bootstrap work directory."""

    work_dir: Path
    bootstrap_config_path: Path
    control_path: Path
    run_state_path: Path
    history_jsonl_path: Path
    latest_status_path: Path
    launcher_pid_path: Path
    launcher_process_state_path: Path
    launcher_stdout_log_path: Path
    launcher_stderr_log_path: Path
    tree_snapshot_dir: Path
    runtime_checkpoint_dir: Path
    rows_dir: Path
    model_dir: Path

    @classmethod
    def from_work_dir(
        cls,
        work_dir: str | Path,
    ) -> MorpionBootstrapPaths:
        """Build canonical bootstrap paths for one work directory."""
        root = Path(work_dir).resolve()
        return cls(
            work_dir=root,
            bootstrap_config_path=root / "bootstrap_config.json",
            control_path=root / "control.json",
            run_state_path=root / "run_state.json",
            history_jsonl_path=root / "history.jsonl",
            latest_status_path=root / "latest_status.json",
            launcher_pid_path=root / "launcher.pid",
            launcher_process_state_path=root / "launcher_process_state.json",
            launcher_stdout_log_path=root / "launcher.out.log",
            launcher_stderr_log_path=root / "launcher.err.log",
            tree_snapshot_dir=root / "tree_exports",
            runtime_checkpoint_dir=root / "search_checkpoints",
            rows_dir=root / "rows",
            model_dir=root / "models",
        )

    def ensure_directories(self) -> None:
        """Create the canonical bootstrap directories if they do not exist."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.tree_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def tree_snapshot_path_for_generation(self, generation: int) -> Path:
        """Return the tree export path for one saved generation."""
        return self.tree_snapshot_dir / f"generation_{generation:06d}.json"

    def rows_path_for_generation(self, generation: int) -> Path:
        """Return the raw Morpion rows path for one saved generation."""
        return self.rows_dir / f"generation_{generation:06d}.json"

    def runtime_checkpoint_path_for_generation(self, generation: int) -> Path:
        """Return the runtime checkpoint path for one saved generation."""
        return self.runtime_checkpoint_dir / f"generation_{generation:06d}.json"

    def model_generation_dir_for_generation(self, generation: int) -> Path:
        """Return the model root directory for one saved generation."""
        return self.model_dir / f"generation_{generation:06d}"

    def model_bundle_path_for_generation(
        self,
        generation: int,
        evaluator_name: str,
    ) -> Path:
        """Return the model bundle directory for one evaluator and generation."""
        return self.model_generation_dir_for_generation(generation) / evaluator_name

    def history_paths(self) -> MorpionBootstrapHistoryPaths:
        """Return the canonical bootstrap history artifact paths."""
        return MorpionBootstrapHistoryPaths(
            work_dir=self.work_dir,
            history_jsonl_path=self.history_jsonl_path,
            latest_status_path=self.latest_status_path,
        )

    def relative_to_work_dir(self, path: str | Path) -> str:
        """Return one persisted path relative to ``work_dir`` or fail clearly."""
        raw_path = Path(path)
        if not raw_path.is_absolute():
            return raw_path.as_posix()
        try:
            return raw_path.relative_to(self.work_dir).as_posix()
        except ValueError as exc:
            raise InvalidBootstrapArtifactPathError(raw_path, self.work_dir) from exc

    def resolve_work_dir_path(self, path: str | Path | None) -> Path | None:
        """Resolve one possibly-relative persisted path against ``work_dir``."""
        if path is None:
            return None
        raw_path = Path(path)
        if raw_path.is_absolute():
            return raw_path
        return self.work_dir / raw_path


class MorpionSearchRunner(Protocol):
    """Thin search-runner boundary for the Morpion bootstrap loop."""

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
    ) -> None:
        """Load existing search state or initialize a fresh one."""
        ...

    def grow(self, max_growth_steps: int) -> None:
        """Grow the underlying search state by a bounded number of steps."""
        ...

    def export_training_tree_snapshot(
        self,
        output_path: str | Path,
    ) -> None:
        """Persist a training-grade tree snapshot to ``output_path``."""
        ...

    def current_tree_size(self) -> int:
        """Return the current size of the search tree."""
        ...


def build_bootstrap_event(
    *,
    cycle_index: int,
    generation: int,
    timestamp_utc: str,
    tree_status: MorpionBootstrapTreeStatus,
    tree_snapshot_path: str | None,
    rows_path: str | None,
    dataset_num_rows: int | None,
    dataset_num_samples: int | None,
    training_triggered: bool,
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics] | None = None,
    model_bundle_paths: Mapping[str, str] | None = None,
    record_status: MorpionBootstrapRecordStatus | None = None,
    event_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MorpionBootstrapEvent:
    """Build one structured bootstrap history event from cycle outputs."""
    return MorpionBootstrapEvent(
        event_id=f"cycle_{cycle_index:06d}" if event_id is None else event_id,
        cycle_index=cycle_index,
        generation=generation,
        timestamp_utc=timestamp_utc,
        tree=tree_status,
        dataset=MorpionBootstrapDatasetStatus(
            num_rows=dataset_num_rows,
            num_samples=dataset_num_samples,
        ),
        training=MorpionBootstrapTrainingStatus(triggered=training_triggered),
        record=default_morpion_record_status()
        if record_status is None
        else record_status,
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path=tree_snapshot_path,
            rows_path=rows_path,
            model_bundle_paths=dict(model_bundle_paths or {}),
        ),
        evaluators=dict(evaluator_metrics or {}),
        metadata=dict(metadata or {}),
    )


def should_save_progress(
    *,
    current_tree_size: int,
    tree_size_at_last_save: int,
    now_unix_s: float,
    last_save_unix_s: float | None,
    save_after_tree_growth_factor: float,
    save_after_seconds: float,
) -> bool:
    """Return whether the bootstrap loop should checkpoint and retrain now.

    The first cycle saves immediately when ``last_save_unix_s`` is ``None``.
    After that, saving triggers when the tree has grown enough since the last
    save or when enough wall-clock time has elapsed.
    """
    if last_save_unix_s is None:
        return True
    if (
        tree_size_at_last_save > 0
        and current_tree_size >= tree_size_at_last_save * save_after_tree_growth_factor
    ):
        return True
    return now_unix_s - last_save_unix_s >= save_after_seconds


def run_one_bootstrap_cycle(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    control: MorpionBootstrapControl | None = None,
    bootstrap_config: MorpionBootstrapConfig | None = None,
    now_unix_s: float | None = None,
) -> MorpionBootstrapRunState:
    """Run one grow/export/train/save bootstrap cycle.

    This helper calls ``runner.load_or_create(...)`` on every cycle, so runner
    implementations should support reload/restart-style semantics from the
    latest saved artifacts.
    """
    paths.ensure_directories()
    resolved_control = MorpionBootstrapControl() if control is None else control
    resolved_bootstrap_config = (
        bootstrap_config_from_args(args)
        if bootstrap_config is None
        else bootstrap_config
    )
    effective_runtime_config = effective_runtime_config_from_config_and_control(
        resolved_bootstrap_config,
        resolved_control,
    )
    previous_effective_runtime_config = _previous_effective_runtime_config(
        run_state.metadata,
        resolved_bootstrap_config=resolved_bootstrap_config,
    )
    _validate_runtime_reconfiguration(
        previous_effective_runtime_config=previous_effective_runtime_config,
        effective_runtime_config=effective_runtime_config,
        cycle_index=run_state.cycle_index,
    )
    resolved_evaluators_config = args.resolved_evaluators_config()
    _validate_forced_evaluator(
        force_evaluator=resolved_control.force_evaluator,
        evaluator_names=resolved_evaluators_config.evaluators,
    )
    history_recorder = MorpionBootstrapHistoryRecorder(paths.history_paths())
    resolved_active_model = _resolve_active_model_bundle(
        paths=paths,
        latest_model_bundle_paths=run_state.latest_model_bundle_paths,
        active_evaluator_name=run_state.active_evaluator_name,
        force_evaluator=resolved_control.force_evaluator,
    )
    restore_tree_path = _resolve_runtime_restore_path(paths=paths, run_state=run_state)

    runner.load_or_create(
        restore_tree_path,
        resolved_active_model.model_bundle_path,
        effective_runtime_config,
    )
    runner.grow(args.max_growth_steps_per_cycle)
    current_tree_size = runner.current_tree_size()
    tree_status = _resolve_tree_status(
        runner,
        current_tree_size=current_tree_size,
    )
    current_time = time.time() if now_unix_s is None else now_unix_s
    cycle_index = run_state.cycle_index + 1
    timestamp_utc = _timestamp_utc_from_unix_s(current_time)

    if not should_save_progress(
        current_tree_size=current_tree_size,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        now_unix_s=current_time,
        last_save_unix_s=run_state.last_save_unix_s,
        save_after_tree_growth_factor=args.save_after_tree_growth_factor,
        save_after_seconds=args.save_after_seconds,
    ):
        next_run_state = MorpionBootstrapRunState(
            generation=run_state.generation,
            cycle_index=cycle_index,
            latest_tree_snapshot_path=run_state.latest_tree_snapshot_path,
            latest_rows_path=run_state.latest_rows_path,
            latest_model_bundle_paths=None
            if run_state.latest_model_bundle_paths is None
            else dict(run_state.latest_model_bundle_paths),
            active_evaluator_name=resolved_active_model.active_evaluator_name,
            tree_size_at_last_save=run_state.tree_size_at_last_save,
            last_save_unix_s=run_state.last_save_unix_s,
            latest_record_status=run_state.latest_record_status,
            metadata=_next_metadata(
                run_state.metadata,
                relative_runtime_checkpoint_path=None,
                control=resolved_control,
                effective_runtime_config=effective_runtime_config,
            ),
        )
        history_recorder.record(
            build_bootstrap_event(
                cycle_index=cycle_index,
                generation=next_run_state.generation,
                timestamp_utc=timestamp_utc,
                tree_status=tree_status,
                tree_snapshot_path=None,
                rows_path=None,
                dataset_num_rows=None,
                dataset_num_samples=None,
                training_triggered=False,
                record_status=resolve_record_status_for_cycle(
                    snapshot=None,
                    previous_record_status=run_state.latest_record_status,
                ),
                metadata=_build_event_metadata(
                    active_evaluator_name=next_run_state.active_evaluator_name,
                    config_hash=_bootstrap_config_hash_from_metadata(run_state.metadata),
                    forced_evaluator=resolved_control.force_evaluator,
                    runtime_control=resolved_control.runtime,
                    effective_runtime_config=effective_runtime_config,
                ),
            )
        )
        return next_run_state

    generation = run_state.generation + 1
    tree_snapshot_path = paths.tree_snapshot_path_for_generation(generation)
    runtime_checkpoint_path = paths.runtime_checkpoint_path_for_generation(generation)
    rows_path = paths.rows_path_for_generation(generation)

    relative_runtime_checkpoint_path: str | None = None
    save_checkpoint = getattr(runner, "save_checkpoint", None)
    if callable(save_checkpoint):
        save_checkpoint(runtime_checkpoint_path)
        relative_runtime_checkpoint_path = paths.relative_to_work_dir(
            runtime_checkpoint_path
        )

    runner.export_training_tree_snapshot(tree_snapshot_path)
    snapshot = load_training_tree_snapshot(tree_snapshot_path)
    rows = training_tree_snapshot_to_morpion_supervised_rows(
        snapshot,
        require_exact_or_terminal=args.require_exact_or_terminal,
        min_depth=args.min_depth,
        min_visit_count=args.min_visit_count,
        max_rows=args.max_rows,
        use_backed_up_value=args.use_backed_up_value,
        metadata={"bootstrap_generation": generation},
    )
    save_morpion_supervised_rows(rows, rows_path)
    num_rows = len(rows.rows)
    record_status = resolve_record_status_for_cycle(
        snapshot=snapshot,
        previous_record_status=run_state.latest_record_status,
    )

    relative_tree_snapshot_path = paths.relative_to_work_dir(tree_snapshot_path)
    relative_rows_path = paths.relative_to_work_dir(rows_path)

    if num_rows == 0:
        preserved_model_bundle_paths = (
            None
            if run_state.latest_model_bundle_paths is None
            else dict(run_state.latest_model_bundle_paths)
        )
        next_metadata = _next_metadata(
            run_state.metadata,
            relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
            control=resolved_control,
            effective_runtime_config=effective_runtime_config,
            training_skipped_reason=EMPTY_DATASET_TRAINING_SKIPPED_REASON,
        )
        next_run_state = MorpionBootstrapRunState(
            generation=generation,
            cycle_index=cycle_index,
            latest_tree_snapshot_path=relative_tree_snapshot_path,
            latest_rows_path=relative_rows_path,
            latest_model_bundle_paths=preserved_model_bundle_paths,
            active_evaluator_name=run_state.active_evaluator_name,
            tree_size_at_last_save=current_tree_size,
            last_save_unix_s=current_time,
            latest_record_status=record_status,
            metadata=next_metadata,
        )
        history_recorder.record(
            build_bootstrap_event(
                cycle_index=cycle_index,
                generation=next_run_state.generation,
                timestamp_utc=timestamp_utc,
                tree_status=tree_status,
                tree_snapshot_path=relative_tree_snapshot_path,
                rows_path=relative_rows_path,
                dataset_num_rows=num_rows,
                dataset_num_samples=num_rows,
                training_triggered=False,
                record_status=record_status,
                metadata=_build_event_metadata(
                    active_evaluator_name=next_run_state.active_evaluator_name,
                    config_hash=_bootstrap_config_hash_from_metadata(run_state.metadata),
                    forced_evaluator=resolved_control.force_evaluator,
                    runtime_control=resolved_control.runtime,
                    effective_runtime_config=effective_runtime_config,
                    training_skipped_reason=EMPTY_DATASET_TRAINING_SKIPPED_REASON,
                ),
            )
        )
        return next_run_state

    evaluator_metrics: dict[str, MorpionEvaluatorMetrics] = {}
    model_bundle_paths: dict[str, str] = {}
    for evaluator_name, spec in resolved_evaluators_config.evaluators.items():
        model_bundle_path = paths.model_bundle_path_for_generation(
            generation, evaluator_name
        )
        _model, metrics = train_morpion_regressor(
            MorpionTrainingArgs(
                dataset_file=rows_path,
                output_dir=model_bundle_path,
                batch_size=spec.batch_size,
                num_epochs=spec.num_epochs,
                learning_rate=spec.learning_rate,
                shuffle=args.shuffle,
                model_kind=spec.model_type,
                feature_subset_name=spec.feature_subset_name,
                feature_names=spec.feature_names,
                hidden_sizes=spec.hidden_sizes,
            )
        )
        evaluator_metrics[evaluator_name] = MorpionEvaluatorMetrics(
            final_loss=float(metrics["final_loss"]),
            num_epochs=int(metrics["num_epochs"]),
            num_samples=int(metrics["num_samples"]),
        )
        model_bundle_paths[evaluator_name] = paths.relative_to_work_dir(
            model_bundle_path
        )
    selected_evaluator_name = _select_active_evaluator_name(
        evaluator_metrics=evaluator_metrics,
        force_evaluator=resolved_control.force_evaluator,
    )

    next_metadata = _next_metadata(
        run_state.metadata,
        relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        control=resolved_control,
        effective_runtime_config=effective_runtime_config,
    )
    next_run_state = MorpionBootstrapRunState(
        generation=generation,
        cycle_index=cycle_index,
        latest_tree_snapshot_path=relative_tree_snapshot_path,
        latest_rows_path=relative_rows_path,
        latest_model_bundle_paths=model_bundle_paths,
        active_evaluator_name=selected_evaluator_name,
        tree_size_at_last_save=current_tree_size,
        last_save_unix_s=current_time,
        latest_record_status=record_status,
        metadata=next_metadata,
    )
    history_recorder.record(
        build_bootstrap_event(
            cycle_index=cycle_index,
            generation=next_run_state.generation,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            tree_snapshot_path=relative_tree_snapshot_path,
            rows_path=relative_rows_path,
            dataset_num_rows=len(rows.rows),
            dataset_num_samples=len(rows.rows),
            training_triggered=True,
            evaluator_metrics=evaluator_metrics,
            model_bundle_paths=model_bundle_paths,
            record_status=record_status,
            metadata=_build_event_metadata(
                active_evaluator_name=selected_evaluator_name,
                selected_evaluator_name=selected_evaluator_name,
                config_hash=_bootstrap_config_hash_from_metadata(run_state.metadata),
                forced_evaluator=resolved_control.force_evaluator,
                runtime_control=resolved_control.runtime,
                effective_runtime_config=effective_runtime_config,
            ),
        )
    )
    return next_run_state


def run_morpion_bootstrap_loop(
    args: MorpionBootstrapArgs,
    runner: MorpionSearchRunner,
    *,
    max_cycles: int | None = None,
) -> MorpionBootstrapRunState:
    """Run the Morpion bootstrap loop for a bounded number of cycles or forever."""
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()

    current_config = bootstrap_config_from_args(args)
    if paths.bootstrap_config_path.is_file():
        persisted_config = load_bootstrap_config(paths.bootstrap_config_path)
        validate_bootstrap_config_change(persisted_config, current_config)
    save_bootstrap_config(current_config, paths.bootstrap_config_path)
    config_hash = bootstrap_config_sha256(current_config)

    if paths.run_state_path.is_file():
        run_state = load_bootstrap_run_state(paths.run_state_path)
    else:
        run_state = initialize_bootstrap_run_state()
    run_state = _with_config_hash_metadata(run_state, config_hash=config_hash)

    cycles_run = 0
    while max_cycles is None or cycles_run < max_cycles:
        control = load_bootstrap_control(paths.control_path)
        effective_args = apply_control_to_args(args, control)
        run_state = run_one_bootstrap_cycle(
            args=effective_args,
            paths=paths,
            runner=runner,
            run_state=run_state,
            control=control,
            bootstrap_config=current_config,
        )
        save_bootstrap_run_state(run_state, paths.run_state_path)
        cycles_run += 1

    return run_state


def _timestamp_utc_from_unix_s(timestamp_unix_s: float) -> str:
    """Format one Unix timestamp as an ISO 8601 UTC string."""
    timestamp = datetime.fromtimestamp(timestamp_unix_s, tz=UTC)
    timespec = "seconds" if timestamp.microsecond == 0 else "microseconds"
    return timestamp.isoformat(timespec=timespec).replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ResolvedActiveMorpionModelBundle:
    """Resolved active evaluator identity and bundle path for one cycle."""

    active_evaluator_name: str | None
    model_bundle_path: Path | None


def _resolve_active_model_bundle(
    *,
    paths: MorpionBootstrapPaths,
    latest_model_bundle_paths: Mapping[str, str] | None,
    active_evaluator_name: str | None,
    force_evaluator: str | None = None,
) -> ResolvedActiveMorpionModelBundle:
    """Resolve the active evaluator identity and bundle path for runner bootstrap."""
    if not latest_model_bundle_paths:
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=None,
            model_bundle_path=None,
        )
    if force_evaluator is not None:
        selected_path = latest_model_bundle_paths.get(force_evaluator)
        if selected_path is None:
            raise MissingForcedMorpionEvaluatorBundleError(force_evaluator)
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=force_evaluator,
            model_bundle_path=paths.resolve_work_dir_path(selected_path),
        )
    if active_evaluator_name is not None:
        selected_path = latest_model_bundle_paths.get(active_evaluator_name)
        if selected_path is None:
            raise UnknownActiveMorpionEvaluatorError(active_evaluator_name)
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=active_evaluator_name,
            model_bundle_path=paths.resolve_work_dir_path(selected_path),
        )
    if len(latest_model_bundle_paths) == 1:
        inferred_active_evaluator_name, selected_path = next(
            iter(latest_model_bundle_paths.items())
        )
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=inferred_active_evaluator_name,
            model_bundle_path=paths.resolve_work_dir_path(selected_path),
        )
    raise MissingActiveMorpionEvaluatorError


def select_active_evaluator_name(
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics],
) -> str:
    """Select the active evaluator using the lowest available final loss."""
    selectable_losses = {
        evaluator_name: metrics.final_loss
        for evaluator_name, metrics in evaluator_metrics.items()
        if metrics.final_loss is not None and math.isfinite(metrics.final_loss)
    }
    if not selectable_losses:
        raise NoSelectableMorpionEvaluatorError
    return min(
        selectable_losses,
        key=lambda evaluator_name: selectable_losses[evaluator_name],
    )


def _build_event_metadata(
    *,
    active_evaluator_name: str | None,
    selected_evaluator_name: str | None = None,
    config_hash: str | None = None,
    forced_evaluator: str | None = None,
    runtime_control: object | None = None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Build history metadata describing the active and selected evaluators."""
    metadata = morpion_bootstrap_experiment_metadata()
    if active_evaluator_name is not None:
        metadata["active_evaluator_name"] = active_evaluator_name
    if selected_evaluator_name is not None:
        metadata["selected_evaluator_name"] = selected_evaluator_name
        metadata["selection_policy"] = "lowest_final_loss"
    if config_hash is not None:
        metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] = config_hash
    if forced_evaluator is not None:
        metadata["forced_evaluator"] = forced_evaluator
    if runtime_control is not None:
        metadata[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] = (
            bootstrap_runtime_control_to_dict(runtime_control)
        )
    if effective_runtime_config is not None:
        metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] = (
            effective_runtime_config_to_dict(effective_runtime_config)
        )
        metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY] = (
            effective_runtime_config_sha256(effective_runtime_config)
        )
    if training_skipped_reason is not None:
        metadata[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return metadata


def _resolve_tree_status(
    runner: MorpionSearchRunner,
    *,
    current_tree_size: int,
) -> MorpionBootstrapTreeStatus:
    """Return the best available tree monitoring status for the current runner."""
    current_tree_status = getattr(runner, "current_tree_status", None)
    if callable(current_tree_status):
        raw_status = current_tree_status()
        if isinstance(raw_status, MorpionBootstrapTreeStatus):
            return MorpionBootstrapTreeStatus(
                num_nodes=current_tree_size,
                num_expanded_nodes=raw_status.num_expanded_nodes,
                num_simulations=raw_status.num_simulations,
                root_visit_count=raw_status.root_visit_count,
            )
        if isinstance(raw_status, Mapping):
            return MorpionBootstrapTreeStatus(
                num_nodes=current_tree_size,
                num_expanded_nodes=_optional_tree_int(
                    raw_status.get("num_expanded_nodes"),
                    field_name="num_expanded_nodes",
                ),
                num_simulations=_optional_tree_int(
                    raw_status.get("num_simulations"),
                    field_name="num_simulations",
                ),
                root_visit_count=_optional_tree_int(
                    raw_status.get("root_visit_count"),
                    field_name="root_visit_count",
                ),
            )
        raise TypeError(
            "Morpion bootstrap runner current_tree_status() must return "
            "MorpionBootstrapTreeStatus or a mapping."
        )
    return MorpionBootstrapTreeStatus(num_nodes=current_tree_size)


def _optional_tree_int(value: object, *, field_name: str) -> int | None:
    """Return one optional integer tree-status field or raise clearly."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            f"Morpion bootstrap tree-status field `{field_name}` must be an int or null."
        )
    if isinstance(value, int):
        return value
    raise TypeError(
        f"Morpion bootstrap tree-status field `{field_name}` must be an int or null."
    )


def _resolve_runtime_restore_path(
    *,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
) -> Path | None:
    """Resolve the best available persisted runtime restore path for one cycle."""
    metadata_runtime_checkpoint = run_state.metadata.get(
        RUNTIME_CHECKPOINT_METADATA_KEY
    )
    if isinstance(metadata_runtime_checkpoint, str):
        return paths.resolve_work_dir_path(metadata_runtime_checkpoint)
    return paths.resolve_work_dir_path(run_state.latest_tree_snapshot_path)


def _bootstrap_config_hash_from_metadata(metadata: Mapping[str, object]) -> str | None:
    """Return the persisted bootstrap config hash from one metadata mapping."""
    value = metadata.get(BOOTSTRAP_CONFIG_HASH_METADATA_KEY)
    return value if isinstance(value, str) else None


def _with_config_hash_metadata(
    run_state: MorpionBootstrapRunState,
    *,
    config_hash: str,
) -> MorpionBootstrapRunState:
    """Return one run state with the accepted bootstrap config hash recorded."""
    next_metadata = dict(run_state.metadata)
    next_metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] = config_hash
    return MorpionBootstrapRunState(
        generation=run_state.generation,
        cycle_index=run_state.cycle_index,
        latest_tree_snapshot_path=run_state.latest_tree_snapshot_path,
        latest_rows_path=run_state.latest_rows_path,
        latest_model_bundle_paths=None
        if run_state.latest_model_bundle_paths is None
        else dict(run_state.latest_model_bundle_paths),
        active_evaluator_name=run_state.active_evaluator_name,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        last_save_unix_s=run_state.last_save_unix_s,
        latest_record_status=run_state.latest_record_status,
        metadata=next_metadata,
    )


def _validate_forced_evaluator(
    *,
    force_evaluator: str | None,
    evaluator_names: Mapping[str, MorpionEvaluatorSpec],
) -> None:
    """Validate one optional forced evaluator against the configured evaluator set."""
    if force_evaluator is None:
        return
    if force_evaluator not in evaluator_names:
        raise UnknownForcedMorpionEvaluatorError(force_evaluator)


def _select_active_evaluator_name(
    *,
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics],
    force_evaluator: str | None,
) -> str:
    """Return the forced evaluator when present, else the default auto-selection."""
    if force_evaluator is not None:
        if force_evaluator not in evaluator_metrics:
            raise UnknownForcedMorpionEvaluatorError(force_evaluator)
        return force_evaluator
    return select_active_evaluator_name(evaluator_metrics)


def _next_metadata(
    current_metadata: Mapping[str, object],
    *,
    relative_runtime_checkpoint_path: str | None,
    control: MorpionBootstrapControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Return updated run metadata after one cycle boundary."""
    next_metadata = dict(current_metadata)
    if relative_runtime_checkpoint_path is None:
        next_metadata.pop(RUNTIME_CHECKPOINT_METADATA_KEY, None)
    else:
        next_metadata[RUNTIME_CHECKPOINT_METADATA_KEY] = relative_runtime_checkpoint_path
    next_metadata[BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY] = bootstrap_control_to_dict(
        control
    )
    next_metadata[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] = (
        bootstrap_runtime_control_to_dict(control.runtime)
    )
    next_metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] = (
        effective_runtime_config_to_dict(effective_runtime_config)
    )
    next_metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY] = (
        effective_runtime_config_sha256(effective_runtime_config)
    )
    if training_skipped_reason is None:
        next_metadata.pop(TRAINING_SKIPPED_REASON_METADATA_KEY, None)
    else:
        next_metadata[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return next_metadata


def _previous_effective_runtime_config(
    metadata: Mapping[str, object],
    *,
    resolved_bootstrap_config: MorpionBootstrapConfig,
) -> MorpionBootstrapEffectiveRuntimeConfig | None:
    """Return the last applied runtime config, falling back for legacy run metadata."""
    persisted_runtime = effective_runtime_config_from_metadata(
        metadata.get(BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY)
    )
    if persisted_runtime is not None:
        return persisted_runtime
    runtime_checkpoint_path = metadata.get(RUNTIME_CHECKPOINT_METADATA_KEY)
    if runtime_checkpoint_path is None:
        return None
    return MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=resolved_bootstrap_config.runtime.tree_branch_limit,
    )


def _validate_runtime_reconfiguration(
    *,
    previous_effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    cycle_index: int,
) -> None:
    """Validate that the requested runtime change stays within the supported subset.

    The current Anemone restore path safely supports only non-increasing
    ``tree_branch_limit`` changes on an existing persisted tree. Tightening the
    limit only constrains future growth, while widening would retroactively allow
    expansions that the earlier runtime configuration may have pruned away.
    """
    if previous_effective_runtime_config is None or cycle_index < 0:
        return
    if (
        effective_runtime_config.tree_branch_limit
        > previous_effective_runtime_config.tree_branch_limit
    ):
        raise UnsupportedMorpionRuntimeReconfigurationError(
            previous_tree_branch_limit=
            previous_effective_runtime_config.tree_branch_limit,
            requested_tree_branch_limit=effective_runtime_config.tree_branch_limit,
        )


__all__ = [
    "EMPTY_DATASET_TRAINING_SKIPPED_REASON",
    "EmptyMorpionEvaluatorsConfigError",
    "InconsistentMorpionEvaluatorSpecNameError",
    "MissingForcedMorpionEvaluatorBundleError",
    "MissingActiveMorpionEvaluatorError",
    "MorpionBootstrapArgs",
    "MorpionBootstrapPaths",
    "MorpionEvaluatorSpec",
    "MorpionEvaluatorsConfig",
    "MorpionSearchRunner",
    "NoSelectableMorpionEvaluatorError",
    "UnsupportedMorpionRuntimeReconfigurationError",
    "UnknownForcedMorpionEvaluatorError",
    "UnknownActiveMorpionEvaluatorError",
    "TRAINING_SKIPPED_REASON_METADATA_KEY",
    "build_bootstrap_event",
    "run_morpion_bootstrap_loop",
    "run_one_bootstrap_cycle",
    "select_active_evaluator_name",
    "should_save_progress",
]
