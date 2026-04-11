"""Tests for the restartable Morpion bootstrap loop."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    save_training_tree_snapshot,
)
from atomheart.games.morpion import MorpionDynamics as AtomMorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec
from torch.utils.data import DataLoader

from chipiron.environments.morpion.bootstrap import (
    MalformedMorpionBootstrapRunStateError,
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    run_morpion_bootstrap_loop,
    run_one_bootstrap_cycle,
    save_bootstrap_run_state,
    should_save_progress,
)
from chipiron.environments.morpion.learning import load_morpion_supervised_rows
from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
)


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_training_snapshot(
    *,
    target_value: float,
    root_node_id: str,
) -> TrainingTreeSnapshot:
    """Build one minimal valid training snapshot for the bootstrap loop."""
    node = TrainingNodeSnapshot(
        node_id=root_node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=False,
        is_exact=True,
        over_event_label=None,
        visit_count=7,
        metadata={"source": "bootstrap-test"},
    )
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=(node,),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


class FakeMorpionSearchRunner:
    """Tiny deterministic runner satisfying the Morpion bootstrap protocol."""

    def __init__(
        self,
        *,
        tree_sizes: tuple[int, ...],
        target_values: tuple[float, ...],
    ) -> None:
        """Initialize the fake runner with per-cycle tree sizes and targets."""
        self._tree_sizes = tree_sizes
        self._target_values = target_values
        self._cycle_index = -1
        self.load_calls: list[tuple[str | None, str | None]] = []
        self.grow_calls: list[int] = []
        self.export_calls: list[str] = []

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
    ) -> None:
        """Record the latest tree/model inputs used to initialize the runner."""
        self.load_calls.append(
            (
                None if tree_snapshot_path is None else str(tree_snapshot_path),
                None if model_bundle_path is None else str(model_bundle_path),
            )
        )

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner to the next predefined tree size."""
        self.grow_calls.append(max_growth_steps)
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def export_training_tree_snapshot(
        self,
        output_path: str | Path,
    ) -> None:
        """Write one real training snapshot to ``output_path``."""
        self.export_calls.append(str(output_path))
        index = max(self._cycle_index, 0)
        snapshot = _make_training_snapshot(
            target_value=self._target_values[index],
            root_node_id=f"node-{index}",
        )
        save_training_tree_snapshot(snapshot, output_path)

    def current_tree_size(self) -> int:
        """Return the current predefined tree size."""
        index = max(self._cycle_index, 0)
        return self._tree_sizes[index]


def test_should_save_progress_helper() -> None:
    """The save trigger should fire on first save, growth, or elapsed time."""
    assert should_save_progress(
        current_tree_size=1,
        tree_size_at_last_save=0,
        now_unix_s=100.0,
        last_save_unix_s=None,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )
    assert should_save_progress(
        current_tree_size=20,
        tree_size_at_last_save=10,
        now_unix_s=120.0,
        last_save_unix_s=100.0,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )
    assert should_save_progress(
        current_tree_size=11,
        tree_size_at_last_save=10,
        now_unix_s=3700.0,
        last_save_unix_s=0.0,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )
    assert not should_save_progress(
        current_tree_size=15,
        tree_size_at_last_save=10,
        now_unix_s=100.0,
        last_save_unix_s=50.0,
        save_after_tree_growth_factor=2.0,
        save_after_seconds=3600.0,
    )


def test_run_state_round_trip(tmp_path: Path) -> None:
    """Persisted bootstrap run state should round-trip cleanly."""
    state = MorpionBootstrapRunState(
        generation=3,
        cycle_index=17,
        latest_tree_snapshot_path="tree_exports/generation_000003.json",
        latest_rows_path="rows/generation_000003.json",
        latest_model_bundle_path="models/generation_000003",
        tree_size_at_last_save=42,
        last_save_unix_s=1234.5,
        metadata={"note": "checkpoint"},
    )
    path = tmp_path / "run_state.json"

    save_bootstrap_run_state(state, path)
    loaded = load_bootstrap_run_state(path)

    assert loaded == state


def test_malformed_run_state_load_fails_loudly(tmp_path: Path) -> None:
    """Malformed persisted run-state payloads should raise clearly."""
    path = tmp_path / "run_state.json"
    path.write_text('{"generation": "oops", "metadata": []}\n', encoding="utf-8")

    with pytest.raises(MalformedMorpionBootstrapRunStateError):
        load_bootstrap_run_state(path)


def test_run_one_cycle_without_save_does_not_train(tmp_path: Path) -> None:
    """A cycle below both save thresholds should skip export and training."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(15,), target_values=(1.25,))
    run_state = MorpionBootstrapRunState(
        generation=2,
        cycle_index=11,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_path=None,
        tree_size_at_last_save=10,
        last_save_unix_s=100.0,
    )

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=run_state,
        now_unix_s=110.0,
    )

    assert next_state.generation == 2
    assert next_state.cycle_index == 12
    assert next_state.latest_tree_snapshot_path is None
    assert next_state.latest_rows_path is None
    assert next_state.latest_model_bundle_path is None
    assert runner.export_calls == []
    assert not paths.tree_snapshot_dir.exists() or list(paths.tree_snapshot_dir.iterdir()) == []
    assert not paths.rows_dir.exists() or list(paths.rows_dir.iterdir()) == []
    assert not paths.model_dir.exists() or list(paths.model_dir.iterdir()) == []


def test_run_one_cycle_with_save_updates_artifacts(tmp_path: Path) -> None:
    """A cycle crossing the save threshold should export rows and train a model."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))
    run_state = initialize_bootstrap_run_state()

    next_state = run_one_bootstrap_cycle(
        args=args,
        paths=paths,
        runner=runner,
        run_state=run_state,
        now_unix_s=200.0,
    )

    assert next_state.generation == 1
    assert next_state.cycle_index == 0
    assert next_state.tree_size_at_last_save == 10
    assert next_state.last_save_unix_s == 200.0
    assert next_state.latest_tree_snapshot_path is not None
    assert next_state.latest_rows_path is not None
    assert next_state.latest_model_bundle_path is not None
    assert next_state.latest_tree_snapshot_path == "tree_exports/generation_000001.json"
    assert next_state.latest_rows_path == "rows/generation_000001.json"
    assert next_state.latest_model_bundle_path == "models/generation_000001"
    assert paths.resolve_work_dir_path(next_state.latest_tree_snapshot_path).is_file()
    assert paths.resolve_work_dir_path(next_state.latest_rows_path).is_file()
    assert paths.resolve_work_dir_path(next_state.latest_model_bundle_path).is_dir()


def test_loop_resumes_from_saved_run_state(tmp_path: Path) -> None:
    """The full loop should resume from persisted state instead of starting over."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=2.0,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    runner = FakeMorpionSearchRunner(
        tree_sizes=(10, 20),
        target_values=(1.25, -0.5),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    first_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    second_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)

    assert (tmp_path / "run_state.json").is_file()
    assert first_state.generation == 1
    assert second_state.generation == 2
    assert first_state.cycle_index == 0
    assert second_state.cycle_index == 1
    assert runner.load_calls[0] == (None, None)
    assert runner.load_calls[1] == (
        str(paths.resolve_work_dir_path(first_state.latest_tree_snapshot_path)),
        str(paths.resolve_work_dir_path(first_state.latest_model_bundle_path)),
    )


def test_saved_rows_come_from_saved_tree_export_path(tmp_path: Path) -> None:
    """Saved rows should reflect the snapshot content written for each generation."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    runner = FakeMorpionSearchRunner(
        tree_sizes=(10, 11),
        target_values=(1.25, -0.5),
    )

    final_state = run_morpion_bootstrap_loop(args, runner, max_cycles=2)
    assert final_state.generation == 2

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    first_rows = load_morpion_supervised_rows(paths.rows_path_for_generation(1))
    second_rows = load_morpion_supervised_rows(paths.rows_path_for_generation(2))

    assert first_rows.rows[0].target_value == 1.25
    assert second_rows.rows[0].target_value == -0.5


def test_saved_dataset_batches_after_cycle(tmp_path: Path) -> None:
    """A saved rows artifact from the loop should load as a trainable dataset."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        num_epochs=1,
        batch_size=1,
        shuffle=False,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))
    state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    assert state.latest_rows_path is not None
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(
            file_name=str(paths.resolve_work_dir_path(state.latest_rows_path))
        )
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    assert batch.get_input_layer().ndim == 2
    assert batch.get_target_value().shape == (1, 1)
    assert batch.get_input_layer().dtype == torch.float32
    assert batch.get_target_value().dtype == torch.float32
