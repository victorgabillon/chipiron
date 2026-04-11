"""Artifact-driven bootstrap loop for Morpion self-training."""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryPaths,
    MorpionBootstrapHistoryRecorder,
    MorpionEvaluatorMetrics,
)
from .run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)


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
        root = Path(work_dir)
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

    def model_bundle_path_for_generation(self, generation: int) -> Path:
        """Return the model bundle directory for one saved generation."""
        return self.model_dir / f"generation_{generation:06d}"

    def history_paths(self) -> MorpionBootstrapHistoryPaths:
        """Return the canonical bootstrap history artifact paths."""
        return MorpionBootstrapHistoryPaths(
            history_jsonl_path=self.history_jsonl_path,
            latest_status_path=self.latest_status_path,
        )


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
    run_state: MorpionBootstrapRunState,
    tree_size: int,
    rows_count: int | None,
    tree_snapshot_path: str | None,
    rows_path: str | None,
    timestamp_unix_s: float,
    evaluator_metrics: tuple[MorpionEvaluatorMetrics, ...] = (),
    current_record: float | int | None = None,
    metadata: dict[str, object] | None = None,
) -> MorpionBootstrapEvent:
    """Build one structured bootstrap history event from cycle outputs."""
    return MorpionBootstrapEvent(
        timestamp_unix_s=timestamp_unix_s,
        generation=run_state.generation,
        tree_size=tree_size,
        tree_size_at_last_save=None
        if run_state.last_save_unix_s is None
        else run_state.tree_size_at_last_save,
        tree_snapshot_path=tree_snapshot_path,
        rows_path=rows_path,
        rows_count=rows_count,
        current_record=current_record,
        evaluators=evaluator_metrics,
        metadata=dict(metadata) if metadata is not None else {},
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
    history_recorder = MorpionBootstrapHistoryRecorder(paths.history_paths())

    runner.load_or_create(
        run_state.latest_tree_snapshot_path,
        run_state.latest_model_bundle_path,
    )
    runner.grow(args.max_growth_steps_per_cycle)
    current_tree_size = runner.current_tree_size()
    current_time = time.time() if now_unix_s is None else now_unix_s

    if not should_save_progress(
        current_tree_size=current_tree_size,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        now_unix_s=current_time,
        last_save_unix_s=run_state.last_save_unix_s,
        save_after_tree_growth_factor=args.save_after_tree_growth_factor,
        save_after_seconds=args.save_after_seconds,
    ):
        history_recorder.record(
            build_bootstrap_event(
                run_state=run_state,
                tree_size=current_tree_size,
                rows_count=None,
                tree_snapshot_path=None,
                rows_path=None,
                timestamp_unix_s=current_time,
            )
        )
        return run_state

    generation = run_state.generation + 1
    tree_snapshot_path = paths.tree_snapshot_path_for_generation(generation)
    rows_path = paths.rows_path_for_generation(generation)
    model_bundle_path = paths.model_bundle_path_for_generation(generation)

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

    _model, metrics = train_morpion_regressor(
        MorpionTrainingArgs(
            dataset_file=rows_path,
            output_dir=model_bundle_path,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            shuffle=args.shuffle,
            model_kind=args.model_kind,
            hidden_dim=args.hidden_dim,
        )
    )

    next_run_state = MorpionBootstrapRunState(
        generation=generation,
        latest_tree_snapshot_path=str(tree_snapshot_path),
        latest_rows_path=str(rows_path),
        latest_model_bundle_path=str(model_bundle_path),
        tree_size_at_last_save=current_tree_size,
        last_save_unix_s=current_time,
        metadata=dict(run_state.metadata),
    )
    evaluator_metrics = (
        MorpionEvaluatorMetrics(
            name="default",
            model_bundle_path=str(model_bundle_path),
            final_loss=float(metrics["final_loss"]),
            num_epochs=int(metrics["num_epochs"]),
            num_samples=int(metrics["num_samples"]),
        ),
    )
    history_recorder.record(
        build_bootstrap_event(
            run_state=next_run_state,
            tree_size=current_tree_size,
            rows_count=len(rows.rows),
            tree_snapshot_path=str(tree_snapshot_path),
            rows_path=str(rows_path),
            timestamp_unix_s=current_time,
            evaluator_metrics=evaluator_metrics,
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


__all__ = [
    "MorpionBootstrapArgs",
    "MorpionBootstrapPaths",
    "MorpionSearchRunner",
    "build_bootstrap_event",
    "run_morpion_bootstrap_loop",
    "run_one_bootstrap_cycle",
    "should_save_progress",
]
