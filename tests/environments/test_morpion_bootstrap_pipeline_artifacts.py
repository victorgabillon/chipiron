"""Tests for Morpion bootstrap pipeline artifact contracts."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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

import chipiron.environments.morpion.bootstrap.cycle_dataset as cycle_dataset_module
import chipiron.environments.morpion.bootstrap.pipeline_artifacts as pipeline_artifacts_module
from chipiron.environments.morpion.bootstrap import (
    EMPTY_DATASET_TRAINING_SKIPPED_REASON,
    TRAINING_SKIPPED_REASON_METADATA_KEY,
    InvalidMorpionPipelineArtifactError,
    MissingMorpionPipelineArtifactError,
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionPipelineActiveModel,
    MorpionPipelineGenerationManifest,
    MorpionPipelineStageClaim,
    MorpionReevaluationCursor,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    delete_reevaluation_cursor,
    delete_reevaluation_patch,
    load_pipeline_active_model,
    load_pipeline_manifest,
    load_pipeline_stage_claim,
    load_reevaluation_cursor,
    load_reevaluation_patch,
    pipeline_manifest_from_dict,
    reevaluation_cursor_from_dict,
    reevaluation_cursor_to_dict,
    reevaluation_patch_from_dict,
    reevaluation_patch_row_from_dict,
    reevaluation_patch_row_to_dict,
    reevaluation_patch_to_dict,
    run_morpion_bootstrap_loop,
    save_pipeline_active_model,
    save_pipeline_manifest,
    save_pipeline_stage_claim,
    save_reevaluation_cursor,
    save_reevaluation_patch,
)
from chipiron.environments.morpion.learning import MorpionSupervisedRows


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
    """Build one minimal valid training snapshot for pipeline tests."""
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
        metadata={"source": "bootstrap-pipeline-test"},
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

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Accept restore inputs without side effects."""
        _ = (
            tree_snapshot_path,
            model_bundle_path,
            effective_runtime_config,
            reevaluate_tree,
        )

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner to the next predefined tree size."""
        _ = max_growth_steps
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Write one real training snapshot to the requested path."""
        index = max(self._cycle_index, 0)
        snapshot = _make_training_snapshot(
            target_value=self._target_values[index],
            root_node_id=f"node-{index}",
        )
        save_training_tree_snapshot(snapshot, output_path)

    def save_checkpoint(self, output_path: str | Path) -> None:
        """Write one checkpoint placeholder so save-branch manifests can point to it."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text('{"checkpoint": true}\n', encoding="utf-8")

    def current_tree_size(self) -> int:
        """Return the current predefined tree size."""
        index = max(self._cycle_index, 0)
        return self._tree_sizes[index]


def _empty_rows_bundle(*, generation: int = 1) -> MorpionSupervisedRows:
    """Return one explicit empty rows bundle for pipeline tests."""
    return MorpionSupervisedRows(
        rows=(),
        metadata={"bootstrap_generation": generation, "num_rows": 0},
    )


def test_pipeline_manifest_roundtrip(tmp_path: Path) -> None:
    """One saved manifest should round-trip through JSON unchanged."""
    manifest = MorpionPipelineGenerationManifest(
        generation=3,
        created_at_utc="2026-04-28T12:00:00Z",
        runtime_checkpoint_path="search_checkpoints/generation_000003.json",
        tree_snapshot_path="tree_exports/generation_000003.json",
        rows_path="rows/generation_000003.json",
        model_bundle_paths={"linear": "models/generation_000003/linear"},
        selected_evaluator_name="linear",
        dataset_status="done",
        training_status="done",
        metadata={"pipeline_mode": "single_process"},
    )
    path = tmp_path / "pipeline" / "generation_000003" / "manifest.json"

    save_pipeline_manifest(manifest, path)
    loaded = load_pipeline_manifest(path)
    text = path.read_text(encoding="utf-8")

    assert loaded == manifest
    assert path.is_file()
    assert text.startswith("{\n  ")
    assert text.endswith("}\n")


def test_pipeline_manifest_default_statuses() -> None:
    """Minimal manifests should default to not-started stage statuses."""
    manifest = MorpionPipelineGenerationManifest(
        generation=0,
        created_at_utc="2026-04-28T12:00:00Z",
    )

    assert manifest.dataset_status == "not_started"
    assert manifest.training_status == "not_started"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"generation": -1, "created_at_utc": "2026-04-28T12:00:00Z"}, ">= 0"),
        ({"generation": True, "created_at_utc": "2026-04-28T12:00:00Z"}, "integer"),
        (
            {
                "generation": 1,
                "created_at_utc": "2026-04-28T12:00:00Z",
                "model_bundle_paths": [],
            },
            "model_bundle_paths",
        ),
        (
            {
                "generation": 1,
                "created_at_utc": "2026-04-28T12:00:00Z",
                "model_bundle_paths": {"linear": 123},
            },
            "model_bundle_paths",
        ),
        (
            {
                "generation": 1,
                "created_at_utc": "2026-04-28T12:00:00Z",
                "dataset_status": "unknown",
            },
            "dataset_status",
        ),
        (
            {
                "generation": 1,
                "created_at_utc": "2026-04-28T12:00:00Z",
                "training_status": "unknown",
            },
            "training_status",
        ),
    ],
)
def test_invalid_manifest_rejects(payload: dict[str, object], match: str) -> None:
    """Malformed manifest payloads should fail loudly."""
    with pytest.raises(InvalidMorpionPipelineArtifactError, match=match):
        pipeline_manifest_from_dict(payload)


def test_pipeline_active_model_roundtrip(tmp_path: Path) -> None:
    """One saved active-model record should round-trip through JSON unchanged."""
    active_model = MorpionPipelineActiveModel(
        generation=3,
        evaluator_name="linear",
        model_bundle_path="models/generation_000003/linear",
        updated_at_utc="2026-04-28T12:00:00Z",
        metadata={"selection_policy": "lowest_final_loss"},
    )
    path = tmp_path / "pipeline" / "active_model.json"

    save_pipeline_active_model(active_model, path)

    assert load_pipeline_active_model(path) == active_model


def test_pipeline_path_helpers_and_directory_creation(tmp_path: Path) -> None:
    """Pipeline path helpers should resolve stable generation-scoped artifact paths."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()

    assert paths.pipeline_dir == tmp_path.resolve() / "pipeline"
    assert (
        paths.pipeline_generation_dir_for_generation(1)
        == tmp_path.resolve() / "pipeline" / "generation_000001"
    )
    assert (
        paths.pipeline_manifest_path_for_generation(1)
        == tmp_path.resolve() / "pipeline" / "generation_000001" / "manifest.json"
    )
    assert (
        paths.pipeline_dataset_status_path_for_generation(1)
        == tmp_path.resolve()
        / "pipeline"
        / "generation_000001"
        / "dataset_status.json"
    )
    assert (
        paths.pipeline_training_status_path_for_generation(1)
        == tmp_path.resolve()
        / "pipeline"
        / "generation_000001"
        / "training_status.json"
    )
    assert (
        paths.pipeline_dataset_claim_path_for_generation(1)
        == tmp_path.resolve()
        / "pipeline"
        / "generation_000001"
        / "dataset_claim.json"
    )
    assert (
        paths.pipeline_training_claim_path_for_generation(1)
        == tmp_path.resolve()
        / "pipeline"
        / "generation_000001"
        / "training_claim.json"
    )
    assert (
        paths.pipeline_active_model_path
        == tmp_path.resolve() / "pipeline" / "active_model.json"
    )
    assert (
        paths.pipeline_reevaluation_patch_path
        == tmp_path.resolve() / "pipeline" / "reevaluation_patch.json"
    )
    assert (
        paths.pipeline_reevaluation_cursor_path
        == tmp_path.resolve() / "pipeline" / "reevaluation_cursor.json"
    )
    assert paths.pipeline_dir.is_dir()


def test_pipeline_stage_claim_roundtrip(tmp_path: Path) -> None:
    """One saved stage claim should round-trip through JSON unchanged."""
    claim = MorpionPipelineStageClaim(
        generation=1,
        stage="dataset",
        claim_id="claim-1",
        claimed_at_utc="2026-04-28T12:00:00Z",
        expires_at_utc="2026-04-28T13:00:00Z",
        owner="worker-a",
        metadata={"entrypoint": "test"},
    )
    path = tmp_path / "pipeline" / "generation_000001" / "dataset_claim.json"

    save_pipeline_stage_claim(claim, path)

    assert load_pipeline_stage_claim(path) == claim


def test_reevaluation_patch_row_roundtrip() -> None:
    """One reevaluation patch row should round-trip through dict form unchanged."""
    row = MorpionReevaluationPatchRow(
        node_id="node-1",
        direct_value=0.25,
        backed_up_value=0.5,
        is_exact=True,
        is_terminal=False,
        metadata={"depth": 3},
    )

    assert reevaluation_patch_row_from_dict(reevaluation_patch_row_to_dict(row)) == row


def test_reevaluation_patch_roundtrip(tmp_path: Path) -> None:
    """One reevaluation patch should round-trip through JSON unchanged."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    row_1 = MorpionReevaluationPatchRow(
        node_id="node-1",
        direct_value=0.25,
        backed_up_value=0.5,
        is_exact=True,
        is_terminal=False,
        metadata={"depth": 3},
    )
    row_2 = MorpionReevaluationPatchRow(
        node_id="node-2",
        direct_value=-0.75,
        backed_up_value=None,
        is_exact=None,
        is_terminal=None,
        metadata={"depth": 4},
    )
    patch = MorpionReevaluationPatch(
        patch_id="patch-1",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=3,
        evaluator_name="default",
        model_bundle_path="models/generation_000003/default",
        rows=(row_1, row_2),
        tree_generation=7,
        start_cursor="node-000000",
        end_cursor="node-009999",
        metadata={"max_nodes": 10000},
    )

    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)

    assert load_reevaluation_patch(paths.pipeline_reevaluation_patch_path) == patch
    assert paths.pipeline_reevaluation_patch_path.is_file()

    delete_reevaluation_patch(paths.pipeline_reevaluation_patch_path)

    assert not paths.pipeline_reevaluation_patch_path.exists()


def test_empty_reevaluation_patch_rows_allowed() -> None:
    """Reevaluation patches should allow explicit empty row batches."""
    patch = MorpionReevaluationPatch(
        patch_id="patch-1",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=3,
        evaluator_name="default",
        model_bundle_path="models/generation_000003/default",
        rows=(),
    )

    loaded = reevaluation_patch_from_dict(reevaluation_patch_to_dict(patch))

    assert loaded == patch
    assert loaded.rows == ()


def test_reevaluation_cursor_roundtrip(tmp_path: Path) -> None:
    """One reevaluation cursor should round-trip through JSON unchanged."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    cursor = MorpionReevaluationCursor(
        evaluator_generation=3,
        evaluator_name="default",
        model_bundle_path="models/generation_000003/default",
        next_node_cursor="node-010000",
        updated_at_utc="2026-04-28T12:05:00Z",
        tree_generation=7,
        completed_full_pass_count=1,
        last_patch_id="patch-1",
        metadata={"policy": "all_nodes"},
    )

    assert reevaluation_cursor_from_dict(reevaluation_cursor_to_dict(cursor)) == cursor

    save_reevaluation_cursor(cursor, paths.pipeline_reevaluation_cursor_path)

    assert load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path) == cursor
    assert paths.pipeline_reevaluation_cursor_path.is_file()

    delete_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)

    assert not paths.pipeline_reevaluation_cursor_path.exists()


def test_invalid_reevaluation_patch_json(tmp_path: Path) -> None:
    """Malformed reevaluation patch JSON should fail loudly."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    paths.pipeline_reevaluation_patch_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(
        InvalidMorpionPipelineArtifactError,
        match=r"reevaluation patch artifact .* not valid JSON",
    ):
        load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)


def test_missing_reevaluation_patch(tmp_path: Path) -> None:
    """Missing reevaluation patch artifacts should fail loudly."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    with pytest.raises(
        MissingMorpionPipelineArtifactError,
        match="reevaluation patch artifact does not exist",
    ):
        load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)


@pytest.mark.parametrize(
    ("builder", "match"),
    [
        (
            lambda: MorpionReevaluationPatchRow(node_id="node-1", direct_value=True),
            "direct_value",
        ),
        (
            lambda: MorpionReevaluationPatchRow(
                node_id="node-1",
                direct_value=float("inf"),
            ),
            "direct_value",
        ),
        (
            lambda: MorpionReevaluationPatchRow(node_id="", direct_value=0.25),
            "node_id",
        ),
        (
            lambda: MorpionReevaluationPatch(
                patch_id="patch-1",
                created_at_utc="2026-04-28T12:00:00Z",
                evaluator_generation=3,
                evaluator_name="default",
                model_bundle_path="models/generation_000003/default",
                rows="not rows",
            ),
            "rows",
        ),
        (
            lambda: MorpionReevaluationCursor(
                evaluator_generation=3,
                evaluator_name="default",
                model_bundle_path="models/generation_000003/default",
                next_node_cursor=None,
                updated_at_utc="2026-04-28T12:05:00Z",
                completed_full_pass_count=-1,
            ),
            "completed_full_pass_count",
        ),
    ],
)
def test_invalid_reevaluation_artifacts_reject(
    builder: Callable[[], object],
    match: str,
) -> None:
    """Malformed reevaluation artifacts should fail with field-specific errors."""
    with pytest.raises(InvalidMorpionPipelineArtifactError, match=match):
        builder()


def test_package_root_reexports_reevaluation_artifacts() -> None:
    """Package root should re-export the reevaluation artifact public API."""
    assert MorpionReevaluationPatch is pipeline_artifacts_module.MorpionReevaluationPatch
    assert save_reevaluation_patch is pipeline_artifacts_module.save_reevaluation_patch
    assert (
        load_reevaluation_cursor
        is pipeline_artifacts_module.load_reevaluation_cursor
    )


def test_single_process_cycle_writes_manifest_and_active_model(tmp_path: Path) -> None:
    """Saved single-process training cycles should mirror pipeline artifacts."""
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))

    run_morpion_bootstrap_loop(
        MorpionBootstrapArgs(
            work_dir=tmp_path,
            max_growth_steps_per_cycle=5,
            save_after_tree_growth_factor=1.0,
            save_after_seconds=0.0,
            num_epochs=1,
            batch_size=1,
            shuffle=False,
        ),
        runner,
        max_cycles=1,
    )

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)
    dataset_status_payload = json.loads(
        paths.pipeline_dataset_status_path_for_generation(1).read_text(encoding="utf-8")
    )
    training_status_payload = json.loads(
        paths.pipeline_training_status_path_for_generation(1).read_text(encoding="utf-8")
    )

    assert manifest.dataset_status == "done"
    assert manifest.training_status == "done"
    assert manifest.tree_snapshot_path == "tree_exports/generation_000001.json"
    assert manifest.rows_path == "rows/generation_000001.json"
    assert manifest.runtime_checkpoint_path == "search_checkpoints/generation_000001.json"
    assert manifest.selected_evaluator_name is not None
    assert (
        manifest.model_bundle_paths[manifest.selected_evaluator_name]
        == active_model.model_bundle_path
    )
    assert active_model.generation == 1
    assert active_model.evaluator_name == manifest.selected_evaluator_name
    assert dataset_status_payload["status"] == "done"
    assert training_status_payload["status"] == "done"


def test_empty_dataset_save_writes_manifest_without_active_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty saved datasets should produce a done/not-started manifest only."""
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.25,))
    monkeypatch.setattr(
        cycle_dataset_module,
        "training_tree_snapshot_to_morpion_supervised_rows",
        lambda *args, **kwargs: _empty_rows_bundle(generation=1),
    )

    run_morpion_bootstrap_loop(
        MorpionBootstrapArgs(
            work_dir=tmp_path,
            max_growth_steps_per_cycle=5,
            save_after_tree_growth_factor=1.0,
            save_after_seconds=0.0,
        ),
        runner,
        max_cycles=1,
    )

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    dataset_status_payload = json.loads(
        paths.pipeline_dataset_status_path_for_generation(1).read_text(encoding="utf-8")
    )
    training_status_payload = json.loads(
        paths.pipeline_training_status_path_for_generation(1).read_text(encoding="utf-8")
    )

    assert manifest.dataset_status == "done"
    assert manifest.training_status == "not_started"
    assert manifest.selected_evaluator_name is None
    assert manifest.model_bundle_paths == {}
    assert manifest.metadata[TRAINING_SKIPPED_REASON_METADATA_KEY] == (
        EMPTY_DATASET_TRAINING_SKIPPED_REASON
    )
    assert dataset_status_payload["status"] == "done"
    assert training_status_payload["status"] == "not_started"
    assert not paths.pipeline_active_model_path.exists()
