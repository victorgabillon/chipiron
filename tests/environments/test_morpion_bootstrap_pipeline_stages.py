"""Tests for Morpion bootstrap Phase 3 pipeline stage entrypoints."""
# ruff: noqa: E402

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

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
import chipiron.environments.morpion.bootstrap.pipeline_stages as pipeline_stages_module
import chipiron.environments.morpion.bootstrap.search_runner_protocol as search_runner_protocol_module
from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    MorpionPipelineGenerationManifest,
    MorpionSearchRunner,
    PipelineStageAlreadyClaimedError,
    claim_pipeline_stage,
    load_bootstrap_run_state,
    load_pipeline_active_model,
    load_pipeline_manifest,
    run_morpion_bootstrap_experiment,
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
    save_bootstrap_run_state,
    save_pipeline_manifest,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    save_morpion_supervised_rows,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


class FakeMorpionSearchRunner:
    """Tiny deterministic runner satisfying the bootstrap stage protocol."""

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

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Record restore inputs without mutating external state."""
        del effective_runtime_config, reevaluate_tree
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

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Write one real training snapshot to the requested path."""
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
    return codec.dump_state_ref(next_state)


def _make_training_snapshot(
    *,
    target_value: float,
    root_node_id: str,
) -> TrainingTreeSnapshot:
    """Build one minimal valid training snapshot for pipeline stage tests."""
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
        metadata={"source": "bootstrap-pipeline-stage-test"},
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
                metadata={"source": "pipeline-training-test"},
            ),
        ),
        metadata={"bootstrap_generation": 1, "num_rows": 1},
    )


def _artifact_pipeline_args(work_dir: Path) -> MorpionBootstrapArgs:
    """Build one small artifact-pipeline arg set for stage tests."""
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


def _unexpected_full_loop_error() -> AssertionError:
    """Build the assertion used when growth dispatch falls into the full loop."""
    return AssertionError("full loop should not run for artifact-pipeline growth")


def test_pipeline_growth_stage_writes_growth_only_manifest(tmp_path: Path) -> None:
    """Growth stage should export only checkpoint/tree artifacts and manifest state."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))

    run_state = run_pipeline_growth_stage(
        _artifact_pipeline_args(tmp_path),
        runner,
        max_cycles=1,
    )
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))

    assert run_state.generation == 1
    assert manifest.tree_snapshot_path == "tree_exports/generation_000001.json"
    assert manifest.runtime_checkpoint_path == "search_checkpoints/generation_000001.json"
    assert manifest.rows_path is None
    assert manifest.dataset_status == "not_started"
    assert manifest.training_status == "not_started"
    assert manifest.model_bundle_paths == {}
    assert manifest.selected_evaluator_name is None
    assert not paths.rows_path_for_generation(1).exists()
    assert not paths.pipeline_active_model_path.exists()
    assert not paths.model_generation_dir_for_generation(1).exists()


def test_pipeline_growth_stage_then_dataset_then_training(tmp_path: Path) -> None:
    """Staged growth, dataset, and training should hand off purely through artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    args = _artifact_pipeline_args(tmp_path)

    run_pipeline_growth_stage(args, runner, max_cycles=1)
    manifest_after_dataset = run_pipeline_dataset_stage(args, generation=1)
    manifest_after_training = run_pipeline_training_stage(args, generation=1)

    assert manifest_after_dataset.dataset_status == "done"
    assert manifest_after_dataset.rows_path == "rows/generation_000001.json"
    assert manifest_after_dataset.training_status == "not_started"
    assert paths.rows_path_for_generation(1).is_file()
    assert manifest_after_training.training_status == "done"
    assert manifest_after_training.selected_evaluator_name is not None
    assert paths.pipeline_active_model_path.is_file()


def test_pipeline_growth_stage_no_save_only_advances_cycle(tmp_path: Path) -> None:
    """No-save growth cycles should update run state without writing a new manifest."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=3,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=100,
            last_save_unix_s=time.time(),
        ),
        paths.run_state_path,
    )

    runner = FakeMorpionSearchRunner(tree_sizes=(101,), target_values=(1.0,))
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=10.0,
        save_after_seconds=1_000_000.0,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
    )

    run_pipeline_growth_stage(args, runner, max_cycles=1)

    persisted_state = load_bootstrap_run_state(paths.run_state_path)
    assert persisted_state.cycle_index == 4
    assert persisted_state.generation == 1
    assert not paths.pipeline_manifest_path_for_generation(2).exists()


def test_dataset_stage_extracts_rows_from_manifest_tree_snapshot(tmp_path: Path) -> None:
    """Dataset stage should extract rows and mark the manifest done."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    snapshot_path = paths.tree_snapshot_path_for_generation(1)
    save_training_tree_snapshot(
        _make_training_snapshot(target_value=1.25, root_node_id="node-0"),
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

    manifest = run_pipeline_dataset_stage(_artifact_pipeline_args(tmp_path), generation=1)

    assert manifest.rows_path == "rows/generation_000001.json"
    assert paths.rows_path_for_generation(1).is_file()
    assert manifest.dataset_status == "done"
    assert manifest.training_status == "not_started"
    assert not paths.pipeline_dataset_claim_path_for_generation(1).exists()


def test_dataset_stage_blocked_by_active_claim(tmp_path: Path) -> None:
    """Dataset stage should not mutate status when another worker owns the claim."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    snapshot_path = paths.tree_snapshot_path_for_generation(1)
    save_training_tree_snapshot(
        _make_training_snapshot(target_value=1.25, root_node_id="node-0"),
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
    claim_pipeline_stage(
        generation=1,
        stage="dataset",
        claim_path=paths.pipeline_dataset_claim_path_for_generation(1),
        claim_id="first",
        owner="worker-a",
    )

    with pytest.raises(PipelineStageAlreadyClaimedError, match="claim_id=first"):
        run_pipeline_dataset_stage(_artifact_pipeline_args(tmp_path), generation=1)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert manifest.dataset_status == "not_started"


def test_dataset_stage_marks_failed_on_exception(tmp_path: Path) -> None:
    """Dataset stage should mark the manifest failed before re-raising."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            tree_snapshot_path="tree_exports/generation_000001.json",
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"Pipeline tree snapshot does not exist: .*generation_000001.json",
    ):
        run_pipeline_dataset_stage(_artifact_pipeline_args(tmp_path), generation=1)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert manifest.dataset_status == "failed"
    assert not paths.pipeline_dataset_claim_path_for_generation(1).exists()


def test_training_stage_trains_and_updates_active_model(tmp_path: Path) -> None:
    """Training stage should train, select, and publish an active model."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows_path = paths.rows_path_for_generation(1)
    save_morpion_supervised_rows(_make_rows(), rows_path)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )

    manifest = run_pipeline_training_stage(_artifact_pipeline_args(tmp_path), generation=1)
    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)

    assert manifest.training_status == "done"
    assert manifest.selected_evaluator_name is not None
    assert paths.pipeline_active_model_path.is_file()
    assert active_model.evaluator_name == manifest.selected_evaluator_name
    assert (
        active_model.model_bundle_path
        == manifest.model_bundle_paths[manifest.selected_evaluator_name]
    )
    assert not paths.pipeline_training_claim_path_for_generation(1).exists()


def test_training_stage_requires_done_dataset(tmp_path: Path) -> None:
    """Training stage should reject manifests whose dataset stage is incomplete."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )

    with pytest.raises(ValueError, match="dataset_status == 'done'"):
        run_pipeline_training_stage(_artifact_pipeline_args(tmp_path), generation=1)


def test_training_stage_missing_rows_reports_path(tmp_path: Path) -> None:
    """Training stage should surface the missing rows path in its file error."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path="rows/generation_000001.json",
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"Pipeline rows file does not exist: .*generation_000001.json",
    ):
        run_pipeline_training_stage(_artifact_pipeline_args(tmp_path), generation=1)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert manifest.training_status == "failed"
    assert not paths.pipeline_training_claim_path_for_generation(1).exists()


def test_training_stage_blocked_by_active_claim(tmp_path: Path) -> None:
    """Training stage should not mutate status when another worker owns the claim."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows_path = paths.rows_path_for_generation(1)
    save_morpion_supervised_rows(_make_rows(), rows_path)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )
    claim_pipeline_stage(
        generation=1,
        stage="training",
        claim_path=paths.pipeline_training_claim_path_for_generation(1),
        claim_id="first",
        owner="worker-a",
    )

    with pytest.raises(PipelineStageAlreadyClaimedError, match="claim_id=first"):
        run_pipeline_training_stage(_artifact_pipeline_args(tmp_path), generation=1)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert manifest.training_status == "not_started"


def test_launcher_dispatches_dataset_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact-pipeline dataset CLI dispatch should reach the dataset stage."""
    captured: list[tuple[MorpionBootstrapArgs, int]] = []

    def _fake_dataset_stage(
        args: MorpionBootstrapArgs,
        *,
        generation: int,
    ) -> MorpionPipelineGenerationManifest:
        captured.append((args, generation))
        return MorpionPipelineGenerationManifest(
            generation=generation,
            created_at_utc="2026-04-28T12:00:00Z",
        )

    monkeypatch.setattr(launcher_module, "run_pipeline_dataset_stage", _fake_dataset_stage)

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "dataset",
            "--pipeline-generation",
            "1",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )
    run_morpion_bootstrap_experiment(launcher_args)

    assert len(captured) == 1
    assert captured[0][0].pipeline_mode == "artifact_pipeline"
    assert captured[0][1] == 1


def test_launcher_dispatches_growth_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact-pipeline growth CLI dispatch should call the growth stage only."""
    captured: list[tuple[MorpionBootstrapArgs, int]] = []

    def _fake_growth_stage(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_cycles: int = 1,
    ) -> object:
        del runner
        captured.append((args, max_cycles))
        return object()

    def _unexpected_full_loop(*args: object, **kwargs: object) -> object:
        raise _unexpected_full_loop_error()

    monkeypatch.setattr(launcher_module, "run_pipeline_growth_stage", _fake_growth_stage)
    monkeypatch.setattr(launcher_module, "run_morpion_bootstrap_loop", _unexpected_full_loop)

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "growth",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )
    run_morpion_bootstrap_experiment(launcher_args)

    assert len(captured) == 1
    assert captured[0][0].pipeline_mode == "artifact_pipeline"
    assert captured[0][1] == 1


def test_single_process_rejects_non_loop_pipeline_stage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """Single-process CLI mode should reject non-loop pipeline stages."""
    with pytest.raises(SystemExit):
        launcher_module.launcher_args_from_cli(
            [
                "--work-dir",
                str(tmp_path),
                "--pipeline-stage",
                "dataset",
            ]
        )

    assert "only valid with 'loop'" in capsys.readouterr().err


def test_artifact_pipeline_loop_dispatches_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact-pipeline loop CLI dispatch should reach the orchestrator."""
    captured: list[tuple[MorpionBootstrapArgs, int]] = []

    def _fake_orchestrator(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_growth_cycles: int = 1,
    ) -> object:
        del runner
        captured.append((args, max_growth_cycles))
        return object()

    monkeypatch.setattr(
        launcher_module,
        "run_morpion_artifact_pipeline_once",
        _fake_orchestrator,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "loop",
            "--max-cycles",
            "2",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    run_morpion_bootstrap_experiment(launcher_args)

    assert len(captured) == 1
    assert captured[0][0].pipeline_mode == "artifact_pipeline"
    assert captured[0][1] == 2


def test_pipeline_stages_imports_bootstrap_loop_only_for_args() -> None:
    """Pipeline stages should not depend on private bootstrap-loop helpers."""
    source = Path(pipeline_stages_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)

    bootstrap_loop_imports: set[str] = set()
    protocol_imports: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module == "bootstrap_loop":
                bootstrap_loop_imports.update(alias.name for alias in node.names)
            if node.module == "search_runner_protocol":
                protocol_imports.update(alias.name for alias in node.names)

    assert bootstrap_loop_imports <= {"MorpionBootstrapArgs"}
    assert "MorpionSearchRunner" in protocol_imports


def test_package_root_reexports_search_runner_protocol() -> None:
    """Package root should expose the dedicated shared search-runner protocol."""
    assert MorpionSearchRunner is search_runner_protocol_module.MorpionSearchRunner
