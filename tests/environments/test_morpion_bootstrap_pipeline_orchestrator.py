"""Tests for the Morpion Phase 6 artifact-pipeline orchestrator."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

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

import chipiron.environments.morpion.bootstrap.launcher as launcher_module
import chipiron.environments.morpion.bootstrap.pipeline_orchestrator as pipeline_orchestrator_module
from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    MorpionPipelineGenerationManifest,
    MorpionPipelineOrchestratorResult,
    dataset_stage_is_pending,
    initialize_bootstrap_run_state,
    list_pipeline_manifest_generations,
    load_available_pipeline_manifests,
    load_pipeline_active_model,
    load_pipeline_manifest,
    run_morpion_artifact_pipeline_once,
    run_morpion_bootstrap_experiment,
    save_pipeline_manifest,
    training_stage_is_pending,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    save_morpion_supervised_rows,
)


class FakeMorpionSearchRunner:
    """Tiny deterministic runner satisfying the pipeline stage protocol."""

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

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Accept restore inputs without mutating external state."""
        del tree_snapshot_path, model_bundle_path, effective_runtime_config, reevaluate_tree

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner to the next predefined tree size."""
        del max_growth_steps
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Write one real training snapshot to ``output_path``."""
        index = max(self._cycle_index, 0)
        save_training_tree_snapshot(
            _make_training_snapshot(
                target_value=self._target_values[index],
                root_node_id=f"node-{index}",
            ),
            output_path,
        )

    def save_checkpoint(self, output_path: str | Path) -> None:
        """Write one placeholder checkpoint so manifests can point to it."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"checkpoint": true}\n', encoding="utf-8")

    def current_tree_size(self) -> int:
        """Return the current predefined tree size."""
        index = max(self._cycle_index, 0)
        return self._tree_sizes[index]


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
    """Build one minimal valid training snapshot for orchestrator tests."""
    node = TrainingNodeSnapshot(
        node_id=root_node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=True,
        is_exact=True,
        over_event_label=None,
        visit_count=7,
        metadata={"source": "bootstrap-pipeline-orchestrator-test"},
    )
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=(node,),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


def _make_rows() -> MorpionSupervisedRows:
    """Build one minimal valid Morpion supervised-rows dataset."""
    return MorpionSupervisedRows(
        rows=(
            MorpionSupervisedRow(
                node_id="row-1",
                state_ref_payload=_make_morpion_payload(),
                target_value=1.0,
                is_terminal=True,
                is_exact=True,
                depth=2,
                visit_count=3,
                direct_value=0.5,
                metadata={"source": "pipeline-orchestrator-test"},
            ),
        ),
        metadata={"bootstrap_generation": 1, "num_rows": 1},
    )


def _artifact_pipeline_args(work_dir: Path) -> MorpionBootstrapArgs:
    """Build one small artifact-pipeline arg set for orchestrator tests."""
    return MorpionBootstrapArgs(
        work_dir=work_dir,
        pipeline_mode="artifact_pipeline",
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        save_after_seconds=0.0,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
    )


def _write_manifest(path: Path) -> None:
    """Write one placeholder manifest file for discovery-only tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


def test_list_pipeline_manifest_generations_ignores_malformed_entries(
    tmp_path: Path,
) -> None:
    """Manifest discovery should only include exact generation directory names."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _write_manifest(paths.pipeline_manifest_path_for_generation(3))
    _write_manifest(paths.pipeline_manifest_path_for_generation(1))
    (paths.pipeline_dir / "generation_latest").mkdir(parents=True, exist_ok=True)
    (paths.pipeline_dir / "generation_abc").mkdir(parents=True, exist_ok=True)
    (paths.pipeline_dir / "generation_000001_extra").mkdir(parents=True, exist_ok=True)
    (paths.pipeline_dir / "notes.txt").write_text("ignored\n", encoding="utf-8")

    assert list_pipeline_manifest_generations(paths) == (1, 3)


def test_load_available_pipeline_manifests_reads_discovered_manifests(
    tmp_path: Path,
) -> None:
    """Manifest loader should return every discovered persisted manifest by generation."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=3,
            created_at_utc="2026-04-28T12:03:00Z",
            dataset_status="failed",
        ),
        paths.pipeline_manifest_path_for_generation(3),
    )
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:01:00Z",
            training_status="done",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )

    manifests = load_available_pipeline_manifests(paths)

    assert tuple(manifests) == (1, 3)
    assert manifests[1].generation == 1
    assert manifests[1].training_status == "done"
    assert manifests[3].generation == 3
    assert manifests[3].dataset_status == "failed"


def test_pending_stage_predicates_cover_dataset_and_training_cases() -> None:
    """Pending predicates should track exactly the persisted manifest contract."""
    assert dataset_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            tree_snapshot_path="tree_exports/generation_000001.json",
            dataset_status="not_started",
        )
    )
    assert dataset_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            tree_snapshot_path="tree_exports/generation_000001.json",
            dataset_status="failed",
        )
    )
    assert not dataset_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            tree_snapshot_path=None,
            dataset_status="not_started",
        )
    )

    assert training_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path="rows/generation_000001.json",
            dataset_status="done",
            training_status="not_started",
        )
    )
    assert training_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path="rows/generation_000001.json",
            dataset_status="done",
            training_status="failed",
        )
    )
    assert not training_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path="rows/generation_000001.json",
            dataset_status="not_started",
            training_status="not_started",
        )
    )
    assert not training_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=None,
            dataset_status="done",
            training_status="not_started",
        )
    )
    assert not training_stage_is_pending(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path="rows/generation_000001.json",
            dataset_status="done",
            training_status="done",
        )
    )


def test_orchestrator_runs_full_sequential_pipeline_for_new_generation(
    tmp_path: Path,
) -> None:
    """One orchestration pass should grow, extract rows, and train in order."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    args = _artifact_pipeline_args(tmp_path)

    result = run_morpion_artifact_pipeline_once(args, runner, max_growth_cycles=1)
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))

    assert result.growth_run_state is not None
    assert result.dataset_generations == (1,)
    assert result.training_generations == (1,)
    assert manifest.dataset_status == "done"
    assert manifest.training_status == "done"
    assert paths.pipeline_active_model_path.is_file()
    assert load_pipeline_active_model(paths.pipeline_active_model_path).generation == 1


def test_orchestrator_processes_all_pending_generations_in_sorted_order(
    tmp_path: Path,
) -> None:
    """Pending dataset and training generations should be handled without latest-only logic."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()

    snapshot_path = paths.tree_snapshot_path_for_generation(1)
    save_training_tree_snapshot(
        _make_training_snapshot(target_value=1.25, root_node_id="node-1"),
        snapshot_path,
    )
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            tree_snapshot_path=paths.relative_to_work_dir(snapshot_path),
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )

    rows_path = paths.rows_path_for_generation(2)
    save_morpion_supervised_rows(_make_rows(), rows_path)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=2,
            created_at_utc="2026-04-28T12:05:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(2),
    )

    result = run_morpion_artifact_pipeline_once(
        _artifact_pipeline_args(tmp_path),
        FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,)),
        max_growth_cycles=0,
    )

    assert result.growth_run_state is None
    assert result.dataset_generations == (1,)
    # Generation 1 becomes training-pending after its dataset stage completes,
    # so the same orchestration pass should train both generations in order.
    assert result.training_generations == (1, 2)
    assert (
        load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1)).training_status
        == "done"
    )
    assert (
        load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(2)).training_status
        == "done"
    )


def test_single_process_loop_keeps_using_full_bootstrap_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-process launcher mode should not dispatch through the orchestrator."""
    sentinel_runner = object()
    loop_calls: list[tuple[MorpionBootstrapArgs, object, int | None]] = []
    launcher_args = launcher_module.MorpionBootstrapLauncherArgs(
        bootstrap_args=MorpionBootstrapArgs(
            work_dir=tmp_path,
            max_growth_steps_per_cycle=5,
            batch_size=1,
            num_epochs=1,
            shuffle=False,
        ),
        max_cycles=2,
        print_startup_summary=False,
        print_dashboard_hint=False,
    )
    expected_args = launcher_module._resolve_launcher_bootstrap_args(launcher_args)

    def _fake_runner_constructor(runner_args: object) -> object:
        del runner_args
        return sentinel_runner

    def _fake_loop(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_cycles: int | None = None,
    ) -> MorpionBootstrapRunState:
        loop_calls.append((args, runner, max_cycles))
        return initialize_bootstrap_run_state()

    def _unexpected_orchestrator(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_growth_cycles: int = 1,
    ) -> MorpionPipelineOrchestratorResult:
        del args, runner, max_growth_cycles
        raise AssertionError

    monkeypatch.setattr(
        launcher_module,
        "AnemoneMorpionSearchRunner",
        _fake_runner_constructor,
    )
    monkeypatch.setattr(launcher_module, "run_morpion_bootstrap_loop", _fake_loop)
    monkeypatch.setattr(
        launcher_module,
        "run_morpion_artifact_pipeline_once",
        _unexpected_orchestrator,
    )

    result = run_morpion_bootstrap_experiment(launcher_args)

    assert result == initialize_bootstrap_run_state()
    assert loop_calls == [(expected_args, sentinel_runner, 2)]


def test_orchestrator_rejects_negative_max_growth_cycles(tmp_path: Path) -> None:
    """Negative growth-cycle counts should fail clearly instead of silently skipping growth."""
    with pytest.raises(ValueError, match="max_growth_cycles must be >= 0"):
        run_morpion_artifact_pipeline_once(
            _artifact_pipeline_args(tmp_path),
            FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,)),
            max_growth_cycles=-1,
        )


def test_package_root_reexports_pipeline_orchestrator_helpers() -> None:
    """Package root should expose the shared pipeline orchestrator helpers."""
    assert (
        load_available_pipeline_manifests
        is pipeline_orchestrator_module.load_available_pipeline_manifests
    )
    assert (
        run_morpion_artifact_pipeline_once
        is pipeline_orchestrator_module.run_morpion_artifact_pipeline_once
    )
