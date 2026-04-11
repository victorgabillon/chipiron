"""Artifact-driven bootstrap loop for Morpion self-training."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from chipiron.environments.morpion.learning import (
    load_training_tree_snapshot_as_morpion_supervised_rows,
    save_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)

from .history import (
    MorpionBootstrapArtifacts,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryPaths,
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapRecordStatus,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
)
from .run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)


class EmptyMorpionEvaluatorsConfigError(ValueError):
    """Raised when the bootstrap loop is configured with zero evaluators."""

    def __init__(self) -> None:
        """Initialize the empty-evaluators-config error."""
        super().__init__("Morpion bootstrap evaluators_config must contain at least one evaluator.")


class InconsistentMorpionEvaluatorSpecNameError(ValueError):
    """Raised when one evaluator spec name does not match its config key."""

    def __init__(self, key: str, spec_name: str) -> None:
        """Initialize the mismatched-evaluator-name error."""
        super().__init__(
            "Morpion bootstrap evaluator config keys must match spec names, got "
            f"key={key!r} and spec.name={spec_name!r}."
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


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorsConfig:
    """Deterministic collection of evaluator specs for one bootstrap run."""

    evaluators: dict[str, MorpionEvaluatorSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy and validate the evaluator mapping eagerly."""
        copied = dict(self.evaluators)
        if not copied:
            raise EmptyMorpionEvaluatorsConfigError()
        for key, spec in copied.items():
            if key != spec.name:
                raise InconsistentMorpionEvaluatorSpecNameError(key, spec.name)
        object.__setattr__(self, "evaluators", copied)

    def primary_evaluator_name(self) -> str:
        """Return the current primary evaluator for runner bootstrap."""
        if "default" in self.evaluators:
            return "default"
        return next(iter(self.evaluators))


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
    use_backed_up_value: bool = True
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    model_kind: str = "linear"
    hidden_dim: int | None = None
    evaluators_config: MorpionEvaluatorsConfig | None = None

    def resolved_evaluators_config(self) -> MorpionEvaluatorsConfig:
        """Resolve the explicit or legacy single-evaluator config."""
        if self.evaluators_config is not None:
            return self.evaluators_config
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
    run_state_path: Path
    history_jsonl_path: Path
    latest_status_path: Path
    tree_snapshot_dir: Path
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
            run_state_path=root / "run_state.json",
            history_jsonl_path=root / "history.jsonl",
            latest_status_path=root / "latest_status.json",
            tree_snapshot_dir=root / "tree_exports",
            rows_dir=root / "rows",
            model_dir=root / "models",
        )

    def ensure_directories(self) -> None:
        """Create the canonical bootstrap directories if they do not exist."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.tree_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def tree_snapshot_path_for_generation(self, generation: int) -> Path:
        """Return the tree export path for one saved generation."""
        return self.tree_snapshot_dir / f"generation_{generation:06d}.json"

    def rows_path_for_generation(self, generation: int) -> Path:
        """Return the raw Morpion rows path for one saved generation."""
        return self.rows_dir / f"generation_{generation:06d}.json"

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
            raise ValueError(
                f"Bootstrap artifact path {raw_path} must be inside work_dir "
                f"{self.work_dir} to be persisted relatively."
            ) from exc

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
    ) -> None:
        """Load existing search state or initialize a fresh one."""

    def grow(self, max_growth_steps: int) -> None:
        """Grow the underlying search state by a bounded number of steps."""

    def export_training_tree_snapshot(
        self,
        output_path: str | Path,
    ) -> None:
        """Persist a training-grade tree snapshot to ``output_path``."""

    def current_tree_size(self) -> int:
        """Return the current size of the search tree."""


def build_bootstrap_event(
    *,
    cycle_index: int,
    generation: int,
    timestamp_utc: str,
    tree_num_nodes: int,
    tree_snapshot_path: str | None,
    rows_path: str | None,
    dataset_num_rows: int | None,
    dataset_num_samples: int | None,
    training_triggered: bool,
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics] | None = None,
    model_bundle_paths: Mapping[str, str] | None = None,
    current_record: float | int | None = None,
    event_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MorpionBootstrapEvent:
    """Build one structured bootstrap history event from cycle outputs."""
    return MorpionBootstrapEvent(
        event_id=f"cycle_{cycle_index:06d}" if event_id is None else event_id,
        cycle_index=cycle_index,
        generation=generation,
        timestamp_utc=timestamp_utc,
        tree=MorpionBootstrapTreeStatus(num_nodes=tree_num_nodes),
        dataset=MorpionBootstrapDatasetStatus(
            num_rows=dataset_num_rows,
            num_samples=dataset_num_samples,
        ),
        training=MorpionBootstrapTrainingStatus(triggered=training_triggered),
        record=MorpionBootstrapRecordStatus(current=current_record),
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
        and current_tree_size
        >= tree_size_at_last_save * save_after_tree_growth_factor
    ):
        return True
    return now_unix_s - last_save_unix_s >= save_after_seconds


def run_one_bootstrap_cycle(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    now_unix_s: float | None = None,
) -> MorpionBootstrapRunState:
    """Run one grow/export/train/save bootstrap cycle.

    This helper calls ``runner.load_or_create(...)`` on every cycle, so runner
    implementations should support reload/restart-style semantics from the
    latest saved artifacts.
    """
    paths.ensure_directories()
    resolved_evaluators_config = args.resolved_evaluators_config()
    history_recorder = MorpionBootstrapHistoryRecorder(paths.history_paths())

    runner.load_or_create(
        paths.resolve_work_dir_path(run_state.latest_tree_snapshot_path),
        _resolve_primary_model_bundle_path(
            paths=paths,
            latest_model_bundle_paths=run_state.latest_model_bundle_paths,
            evaluators_config=resolved_evaluators_config,
        ),
    )
    runner.grow(args.max_growth_steps_per_cycle)
    current_tree_size = runner.current_tree_size()
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
            tree_size_at_last_save=run_state.tree_size_at_last_save,
            last_save_unix_s=run_state.last_save_unix_s,
            metadata=dict(run_state.metadata),
        )
        history_recorder.record(
            build_bootstrap_event(
                cycle_index=cycle_index,
                generation=next_run_state.generation,
                timestamp_utc=timestamp_utc,
                tree_num_nodes=current_tree_size,
                tree_snapshot_path=None,
                rows_path=None,
                dataset_num_rows=None,
                dataset_num_samples=None,
                training_triggered=False,
            )
        )
        return next_run_state

    generation = run_state.generation + 1
    tree_snapshot_path = paths.tree_snapshot_path_for_generation(generation)
    rows_path = paths.rows_path_for_generation(generation)

    runner.export_training_tree_snapshot(tree_snapshot_path)

    rows = load_training_tree_snapshot_as_morpion_supervised_rows(
        tree_snapshot_path,
        require_exact_or_terminal=args.require_exact_or_terminal,
        min_depth=args.min_depth,
        min_visit_count=args.min_visit_count,
        use_backed_up_value=args.use_backed_up_value,
        metadata={"bootstrap_generation": generation},
    )
    save_morpion_supervised_rows(rows, rows_path)

    evaluator_metrics: dict[str, MorpionEvaluatorMetrics] = {}
    model_bundle_paths: dict[str, str] = {}
    for evaluator_name, spec in resolved_evaluators_config.evaluators.items():
        model_bundle_path = paths.model_bundle_path_for_generation(generation, evaluator_name)
        _model, metrics = train_morpion_regressor(
            MorpionTrainingArgs(
                dataset_file=rows_path,
                output_dir=model_bundle_path,
                batch_size=spec.batch_size,
                num_epochs=spec.num_epochs,
                learning_rate=spec.learning_rate,
                shuffle=args.shuffle,
                model_kind=spec.model_type,
                hidden_sizes=spec.hidden_sizes,
            )
        )
        evaluator_metrics[evaluator_name] = MorpionEvaluatorMetrics(
            final_loss=float(metrics["final_loss"]),
            num_epochs=int(metrics["num_epochs"]),
            num_samples=int(metrics["num_samples"]),
        )
        model_bundle_paths[evaluator_name] = paths.relative_to_work_dir(model_bundle_path)

    relative_tree_snapshot_path = paths.relative_to_work_dir(tree_snapshot_path)
    relative_rows_path = paths.relative_to_work_dir(rows_path)
    next_run_state = MorpionBootstrapRunState(
        generation=generation,
        cycle_index=cycle_index,
        latest_tree_snapshot_path=relative_tree_snapshot_path,
        latest_rows_path=relative_rows_path,
        latest_model_bundle_paths=model_bundle_paths,
        tree_size_at_last_save=current_tree_size,
        last_save_unix_s=current_time,
        metadata=dict(run_state.metadata),
    )
    history_recorder.record(
        build_bootstrap_event(
            cycle_index=cycle_index,
            generation=next_run_state.generation,
            timestamp_utc=timestamp_utc,
            tree_num_nodes=current_tree_size,
            tree_snapshot_path=relative_tree_snapshot_path,
            rows_path=relative_rows_path,
            dataset_num_rows=len(rows.rows),
            dataset_num_samples=len(rows.rows),
            training_triggered=True,
            evaluator_metrics=evaluator_metrics,
            model_bundle_paths=model_bundle_paths,
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

    if paths.run_state_path.is_file():
        run_state = load_bootstrap_run_state(paths.run_state_path)
    else:
        run_state = initialize_bootstrap_run_state()

    cycles_run = 0
    while max_cycles is None or cycles_run < max_cycles:
        run_state = run_one_bootstrap_cycle(
            args=args,
            paths=paths,
            runner=runner,
            run_state=run_state,
        )
        save_bootstrap_run_state(run_state, paths.run_state_path)
        cycles_run += 1

    return run_state


def _timestamp_utc_from_unix_s(timestamp_unix_s: float) -> str:
    """Format one Unix timestamp as an ISO 8601 UTC string."""
    timestamp = datetime.fromtimestamp(timestamp_unix_s, tz=UTC)
    timespec = "seconds" if timestamp.microsecond == 0 else "microseconds"
    return timestamp.isoformat(timespec=timespec).replace("+00:00", "Z")


def _resolve_primary_model_bundle_path(
    *,
    paths: MorpionBootstrapPaths,
    latest_model_bundle_paths: Mapping[str, str] | None,
    evaluators_config: MorpionEvaluatorsConfig,
) -> Path | None:
    """Resolve the current primary evaluator bundle path for runner bootstrap."""
    if not latest_model_bundle_paths:
        return None
    primary_name = evaluators_config.primary_evaluator_name()
    selected_path = latest_model_bundle_paths.get(primary_name)
    if selected_path is None and "default" in latest_model_bundle_paths:
        selected_path = latest_model_bundle_paths["default"]
    if selected_path is None:
        selected_path = next(iter(latest_model_bundle_paths.values()))
    return paths.resolve_work_dir_path(selected_path)


__all__ = [
    "EmptyMorpionEvaluatorsConfigError",
    "InconsistentMorpionEvaluatorSpecNameError",
    "MorpionBootstrapArgs",
    "MorpionBootstrapPaths",
    "MorpionEvaluatorSpec",
    "MorpionEvaluatorsConfig",
    "MorpionSearchRunner",
    "build_bootstrap_event",
    "run_morpion_bootstrap_loop",
    "run_one_bootstrap_cycle",
    "should_save_progress",
]
