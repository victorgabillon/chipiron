"""Tests for Morpion bootstrap Phase 3 pipeline stage entrypoints."""
# ruff: noqa: E402

from __future__ import annotations

import sys
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
from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionPipelineGenerationManifest,
    load_pipeline_active_model,
    load_pipeline_manifest,
    run_morpion_bootstrap_experiment,
    run_pipeline_dataset_stage,
    run_pipeline_training_stage,
    save_pipeline_manifest,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    save_morpion_supervised_rows,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


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

    with pytest.raises(FileNotFoundError):
        run_pipeline_dataset_stage(_artifact_pipeline_args(tmp_path), generation=1)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert manifest.dataset_status == "failed"


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


def test_artifact_pipeline_loop_still_fails_loudly(tmp_path: Path) -> None:
    """Artifact-pipeline loop dispatch should still fail loudly for now."""
    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "loop",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    with pytest.raises(NotImplementedError, match="artifact_pipeline with --pipeline-stage loop"):
        run_morpion_bootstrap_experiment(launcher_args)
