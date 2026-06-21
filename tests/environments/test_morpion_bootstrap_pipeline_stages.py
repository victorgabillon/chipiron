"""Tests for Morpion bootstrap Phase 3 pipeline stage entrypoints."""
# ruff: noqa: E402

from __future__ import annotations

import ast
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

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

import chipiron.environments.morpion.bootstrap.cycle_training as cycle_training_module
import chipiron.environments.morpion.bootstrap.launcher as launcher_module
import chipiron.environments.morpion.bootstrap.pipeline_memory as pipeline_memory_module
import chipiron.environments.morpion.bootstrap.pipeline_stages as pipeline_stages_module
import chipiron.environments.morpion.bootstrap.search_runner_protocol as search_runner_protocol_module
import chipiron.environments.morpion.learning.tree_to_dataset as tree_to_dataset_module
from chipiron.environments.morpion.bootstrap import (
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    AnemoneMorpionSearchRunner,
    IncompatibleStageBootstrapConfigError,
    InvalidMorpionPipelineArtifactError,
    MorpionBootstrapArgs,
    MorpionBootstrapControl,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    MorpionPipelineActiveModel,
    MorpionPipelineEvaluatorTrainingResult,
    MorpionPipelineGenerationManifest,
    MorpionPipelineTrainingCursor,
    MorpionPipelineWorkerResult,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    MorpionReevaluationWorkerResult,
    MorpionSearchRunner,
    PipelineStageAlreadyClaimedError,
    bootstrap_config_from_args,
    claim_pipeline_stage,
    load_bootstrap_config,
    load_bootstrap_run_state,
    load_pipeline_active_model,
    load_pipeline_dataset_status_file,
    load_pipeline_manifest,
    load_pipeline_training_cursor,
    load_pipeline_training_status_file,
    run_morpion_bootstrap_experiment,
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
    save_bootstrap_config,
    save_bootstrap_run_state,
    save_pipeline_active_model,
    save_pipeline_manifest,
    save_pipeline_training_cursor,
    save_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
    iter_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.cycle_training import (
    BootstrapTrainingResult,
    train_and_select_evaluators,
)
from chipiron.environments.morpion.bootstrap.memory_diagnostics import (
    MemoryDiagnostics,
    MemoryDiagnosticsConfig,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    save_morpion_sharded_training_tree_from_live_nodes,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    load_morpion_supervised_rows,
    morpion_supervised_rows_source_from_path,
    save_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from tests.environments.morpion_training_snapshot_helpers import (
    make_training_node_snapshot,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.logging import LogCaptureFixture


class FakeMorpionSearchRunner:
    """Tiny deterministic runner satisfying the bootstrap stage protocol."""

    def __init__(
        self,
        *,
        tree_sizes: tuple[int, ...],
        target_values: tuple[float, ...],
        branch_counts: tuple[int, ...] | None = None,
        patch_apply_result: int | None = None,
        patch_apply_error: Exception | None = None,
    ) -> None:
        """Initialize the fake runner with per-cycle tree sizes and targets."""
        self._tree_sizes = tree_sizes
        self._target_values = target_values
        self._branch_counts = branch_counts
        self._patch_apply_result = patch_apply_result
        self._patch_apply_error = patch_apply_error
        self._cycle_index = -1
        self.load_calls: list[tuple[str | None, str | None]] = []
        self.grow_calls: list[int] = []
        self.checkpoint_calls: list[str] = []
        self.call_order: list[str] = []
        self.received_patches: list[MorpionReevaluationPatch] = []

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
        self.call_order.append("grow")
        self.grow_calls.append(max_growth_steps)
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def apply_reevaluation_patch(self, patch: MorpionReevaluationPatch) -> int | None:
        """Record one reevaluation patch application before growth."""
        self.call_order.append(f"apply_patch:{patch.patch_id}")
        self.received_patches.append(patch)
        if self._patch_apply_error is not None:
            raise self._patch_apply_error
        return self._patch_apply_result

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

    def export_sharded_training_tree_snapshot(
        self,
        output_dir: str | Path,
        *,
        generation: int,
    ) -> Path:
        """Write one sharded training export using the same deterministic snapshot."""
        index = max(self._cycle_index, 0)
        snapshot = _make_training_snapshot(
            target_value=self._target_values[index],
            root_node_id=f"node-{index}",
        )
        live_nodes = tuple(_TrainingSnapshotLiveNode(node) for node in snapshot.nodes)
        manifest_path, _stats = save_morpion_sharded_training_tree_from_live_nodes(
            nodes=live_nodes,
            root_node_id=snapshot.root_node_id,
            output_dir=output_dir,
            generation=generation,
            state_ref_dumper=lambda state: cast("dict[str, object]", state),
            direct_value_extractor=_float_or_none,
            backed_up_value_extractor=_float_or_none,
        )
        return manifest_path

    def save_checkpoint(self, output_path: str | Path) -> None:
        """Write one placeholder checkpoint so manifests can point to it."""
        self.checkpoint_calls.append(str(output_path))
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"checkpoint": true}\n', encoding="utf-8")

    def current_tree_size(self) -> int:
        """Return the current predefined tree size."""
        index = max(self._cycle_index, 0)
        return self._tree_sizes[index]

    def current_tree_branch_count(self) -> int | None:
        """Return the current predefined branch count when provided."""
        if self._branch_counts is None:
            return None
        index = max(self._cycle_index, 0)
        return self._branch_counts[index]


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
    node = make_training_node_snapshot(
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


def _make_rows_with_count(count: int) -> MorpionSupervisedRows:
    """Build a deterministic Morpion supervised-rows dataset of a given size."""
    template = _make_rows().rows[0]
    return MorpionSupervisedRows(
        rows=tuple(
            replace(
                template,
                node_id=f"row-{index + 1}",
                target_value=float(index + 1),
            )
            for index in range(count)
        ),
        metadata={"bootstrap_generation": 1, "num_rows": count},
    )


def test_dataset_row_iterator_matches_materialized_extraction(tmp_path: Path) -> None:
    """Dataset-stage streaming rows should match the materialized extractor."""
    args = _artifact_pipeline_args(tmp_path)
    snapshot = _make_training_snapshot(target_value=1.25, root_node_id="node-0")

    materialized = extract_rows_from_training_snapshot(
        args=args,
        snapshot=snapshot,
        generation=1,
    )
    streamed_rows = tuple(
        iter_rows_from_training_snapshot(
            args=args,
            snapshot=snapshot,
            generation=1,
        )
    )

    assert streamed_rows == materialized.rows


def _fake_training_result(
    paths: MorpionBootstrapPaths,
    *,
    generation: int,
    evaluator_name: str = "linear_5",
) -> BootstrapTrainingResult:
    """Build one minimal training result for mocked training-stage tests."""
    bundle_path = paths.model_bundle_path_for_generation(generation, evaluator_name)
    bundle_path.mkdir(parents=True, exist_ok=True)
    relative_bundle_path = paths.relative_to_work_dir(bundle_path)
    evaluator_result = MorpionPipelineEvaluatorTrainingResult(
        final_loss=0.25,
        elapsed_s=0.1,
        model_bundle_path=relative_bundle_path,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        loss_name="mse",
    )
    return BootstrapTrainingResult(
        generation=generation,
        evaluator_metrics={},
        evaluator_results={evaluator_name: evaluator_result},
        model_bundle_paths={evaluator_name: relative_bundle_path},
        selected_evaluator_name=evaluator_name,
        selection_policy="mock_lowest_loss",
        training_duration_s=0.1,
    )


def _fake_training_result_for_evaluators(
    paths: MorpionBootstrapPaths,
    *,
    generation: int,
    evaluator_names: tuple[str, ...],
    selected_evaluator_name: str,
) -> BootstrapTrainingResult:
    """Build a mocked training result for a named evaluator subset."""
    evaluator_results: dict[str, MorpionPipelineEvaluatorTrainingResult] = {}
    model_bundle_paths: dict[str, str] = {}
    for index, evaluator_name in enumerate(evaluator_names):
        bundle_path = paths.model_bundle_path_for_generation(
            generation, evaluator_name
        )
        bundle_path.mkdir(parents=True, exist_ok=True)
        relative_bundle_path = paths.relative_to_work_dir(bundle_path)
        model_bundle_paths[evaluator_name] = relative_bundle_path
        evaluator_results[evaluator_name] = MorpionPipelineEvaluatorTrainingResult(
            final_loss=0.25 + index,
            elapsed_s=0.1,
            model_bundle_path=relative_bundle_path,
            num_epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            loss_name="mse",
        )
    return BootstrapTrainingResult(
        generation=generation,
        evaluator_metrics={},
        evaluator_results=evaluator_results,
        model_bundle_paths=model_bundle_paths,
        selected_evaluator_name=selected_evaluator_name,
        selection_policy="mock_lowest_loss",
        training_duration_s=0.1,
    )


def _multi_evaluator_config() -> MorpionEvaluatorsConfig:
    """Return one small multi-evaluator config for training-stage tests."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear_5": MorpionEvaluatorSpec(
                name="linear_5",
                model_type="linear",
                hidden_sizes=None,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
            "mlp_5": MorpionEvaluatorSpec(
                name="mlp_5",
                model_type="mlp",
                hidden_sizes=(8, 4),
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
        }
    )


def _make_reevaluation_patch(*, patch_id: str) -> MorpionReevaluationPatch:
    """Build one minimal reevaluation patch artifact for growth-stage tests."""
    return MorpionReevaluationPatch(
        patch_id=patch_id,
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(
            MorpionReevaluationPatchRow(
                node_id="node-a",
                direct_value=1.25,
                metadata={"source": "pipeline-stage-test"},
            ),
        ),
        tree_generation=1,
        start_cursor="node-a",
        end_cursor="node-a",
        metadata={"source": "pipeline-stage-test"},
    )


def _artifact_pipeline_args(work_dir: Path) -> MorpionBootstrapArgs:
    """Build one small artifact-pipeline arg set for stage tests."""
    return MorpionBootstrapArgs(
        work_dir=work_dir,
        pipeline_mode="artifact_pipeline",
        training_export_mode="flat",
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        save_after_seconds=0.0,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
    )


def _prepare_training_stage_input(
    paths: MorpionBootstrapPaths,
    *,
    generation: int = 1,
    rows: MorpionSupervisedRows | None = None,
) -> None:
    """Persist the minimum artifacts needed to enter the training stage."""
    paths.ensure_directories()
    rows_path = paths.rows_path_for_generation(generation)
    save_morpion_supervised_rows(rows if rows is not None else _make_rows(), rows_path)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=generation,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(generation),
    )


def _prepare_training_stage_jsonl_input(
    paths: MorpionBootstrapPaths,
    *,
    generation: int = 1,
    rows: MorpionSupervisedRows | None = None,
) -> None:
    """Persist JSONL row artifacts needed to enter the training stage."""
    paths.ensure_directories()
    rows_bundle = rows if rows is not None else _make_rows()
    rows_path = paths.rows_jsonl_path_for_generation(generation)
    save_morpion_supervised_rows_streaming(
        rows=rows_bundle.rows,
        metadata={**rows_bundle.metadata, "num_rows": len(rows_bundle.rows)},
        path=rows_path,
    )
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=generation,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(generation),
    )


@dataclass(slots=True)
class _TrainingSnapshotLiveNode:
    """Live-node adapter that replays a persisted training snapshot node."""

    node: TrainingNodeSnapshot

    @property
    def id(self) -> str:
        return self.node.node_id

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return self.node.parent_ids

    @property
    def child_ids(self) -> tuple[str, ...]:
        return self.node.child_ids

    @property
    def depth(self) -> int:
        return self.node.depth

    @property
    def state(self) -> dict[str, object]:
        return cast("dict[str, object]", self.node.state_ref_payload)

    @property
    def direct_value(self) -> float | None:
        return self.node.direct_value_scalar

    @property
    def backed_up_value(self) -> float | None:
        return self.node.backed_up_value_scalar

    @property
    def is_terminal(self) -> bool:
        return self.node.is_terminal

    @property
    def is_exact(self) -> bool:
        return self.node.is_exact

    @property
    def visit_count(self) -> int | None:
        return self.node.visit_count

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self.node.metadata)

    @property
    def over_event_label(self) -> str | None:
        return self.node.over_event_label


def _float_or_none(value: object | None) -> float | None:
    """Return float scalars for test live-node adapters."""
    if value is None:
        return None
    return float(cast("int | float", value))


def _unexpected_full_loop_error() -> AssertionError:
    """Build the assertion used when growth dispatch falls into the full loop."""
    return AssertionError("full loop should not run for artifact-pipeline growth")


def _unexpected_reevaluation_runner_error() -> AssertionError:
    """Build the assertion used when reevaluation dispatch builds a runner."""
    return AssertionError("reevaluation stage should not build a runner")


def _unexpected_pipeline_worker_runner_error() -> AssertionError:
    """Build the assertion used when worker dispatch builds a runner."""
    return AssertionError("pipeline worker stage should not build a runner")


def _negative_max_nodes_per_patch_error() -> ValueError:
    """Build the stable worker-style negative batch-size error."""
    return ValueError("max_nodes_per_patch must be >= 0")


def _make_reevaluation_worker_result() -> MorpionReevaluationWorkerResult:
    """Build a representative reevaluation worker result for launcher tests."""
    return MorpionReevaluationWorkerResult(
        patch_written=False,
        reason="test",
        patch_id=None,
        num_rows=0,
        evaluator_generation=None,
        evaluator_name=None,
        start_cursor=None,
        end_cursor=None,
        completed_full_pass_count=None,
    )


def _make_pipeline_worker_result(
    stage: Literal["dataset", "training"],
) -> MorpionPipelineWorkerResult:
    """Build a representative pipeline worker result for launcher tests."""
    return MorpionPipelineWorkerResult(
        stage=stage,
        generation=None,
        ran_stage=False,
        reason="test",
    )


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
    assert manifest.runtime_checkpoint_path == paths.relative_to_work_dir(
        paths.runtime_checkpoint_path_for_generation(1)
    )
    assert manifest.rows_path is None
    assert manifest.dataset_status == "not_started"
    assert manifest.training_status == "not_started"
    assert manifest.model_bundle_paths == {}
    assert manifest.selected_evaluator_name is None
    assert not paths.rows_path_for_generation(1).exists()
    assert not paths.pipeline_active_model_path.exists()
    assert not paths.model_generation_dir_for_generation(1).exists()


def test_pipeline_growth_stage_logs_memory_profile_when_enabled(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """Growth stage should emit opt-in memory profile checkpoints."""
    runner = FakeMorpionSearchRunner(
        tree_sizes=(5,),
        target_values=(1.0,),
        branch_counts=(8,),
    )
    runner.nodes = [
        {
            "metadata": {"index": 1},
            "state_ref_payload": {"state": [1, 2]},
            "children": [2, 3],
        }
    ]
    args = replace(
        _artifact_pipeline_args(tmp_path),
        growth_memory_profile=True,
        growth_memory_profile_top_n=3,
        growth_memory_profile_sample_nodes=1,
    )

    caplog.set_level(logging.INFO)
    run_pipeline_growth_stage(args, runner, max_cycles=1)

    text = caplog.text
    assert "[growth-profile] event=after_checkpoint_load" in text
    assert "[growth-profile] event=before_growth" in text
    assert "[growth-profile] event=after_growth" in text
    assert "[growth-profile] event=before_checkpoint_save" in text
    assert "[growth-profile] event=checkpoint_save_done" in text
    assert "[growth-profile] event=after_checkpoint_save" in text
    assert "node_sample" in text


def test_pipeline_growth_stage_guards_candidate_checkpoint_load_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    """Forecasted low headroom should defer before checkpoint validation loads."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    checkpoint_path = paths.runtime_checkpoint_path_for_generation(1)
    checkpoint_path.write_text("not loaded by this test\n", encoding="utf-8")
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=7,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=5,
            last_save_unix_s=0.0,
            latest_runtime_checkpoint_path=paths.relative_to_work_dir(
                checkpoint_path
            ),
        ),
        paths.run_state_path,
    )
    args = replace(
        _artifact_pipeline_args(tmp_path),
        min_available_ram_mb=5_000,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: 5_500.0)

    caplog.set_level(logging.INFO)
    run_state = run_pipeline_growth_stage(args, runner, max_cycles=1)

    assert run_state.generation == 1
    assert runner.load_calls == []
    assert "[checkpoint] candidate_validate_start" not in caplog.text
    assert "[checkpoint-forecast]" in caplog.text
    assert "action=candidate_checkpoint_load " in caplog.text
    assert "decision=skip" in caplog.text
    assert (
        "[pipeline] growth_skip generation=1 reason=low_available_ram "
        "action=candidate_checkpoint_load_forecast"
    ) in caplog.text


def test_pipeline_growth_stage_loads_candidate_when_forecast_has_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    """Forecast should allow validation when pre-load headroom is sufficient."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    checkpoint_path = paths.runtime_checkpoint_path_for_generation(1)
    checkpoint_runner = AnemoneMorpionSearchRunner()
    checkpoint_runner.load_or_create(None, None)
    checkpoint_runner.grow(1)
    checkpoint_runner.save_checkpoint(checkpoint_path)
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=7,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=5,
            last_save_unix_s=0.0,
            latest_runtime_checkpoint_path=paths.relative_to_work_dir(
                checkpoint_path
            ),
        ),
        paths.run_state_path,
    )
    args = replace(
        _artifact_pipeline_args(tmp_path),
        min_available_ram_mb=5_000,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(6,), target_values=(1.0,))
    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: 6_000.0)

    caplog.set_level(logging.INFO)
    run_pipeline_growth_stage(args, runner, max_cycles=1)

    assert runner.load_calls
    assert "[checkpoint-forecast]" in caplog.text
    assert "decision=run" in caplog.text
    assert "[checkpoint] candidate_validate_start" in caplog.text


def test_pipeline_growth_stage_profiles_candidate_checkpoint_load(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """Growth profiling should include candidate checkpoint load RSS deltas."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    checkpoint_path = paths.runtime_checkpoint_path_for_generation(1)
    checkpoint_runner = AnemoneMorpionSearchRunner()
    checkpoint_runner.load_or_create(None, None)
    checkpoint_runner.grow(1)
    checkpoint_runner.save_checkpoint(checkpoint_path)
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=7,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=5,
            last_save_unix_s=0.0,
            latest_runtime_checkpoint_path=paths.relative_to_work_dir(
                checkpoint_path
            ),
        ),
        paths.run_state_path,
    )
    args = replace(
        _artifact_pipeline_args(tmp_path),
        growth_memory_profile=True,
        growth_memory_profile_top_n=3,
        growth_memory_profile_sample_nodes=1,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(6,), target_values=(1.0,))

    caplog.set_level(logging.INFO)
    run_pipeline_growth_stage(args, runner, max_cycles=1)

    assert "[growth-profile] event=before_candidate_checkpoint_load" in caplog.text
    assert "[growth-profile] event=candidate_checkpoint_load_done" in caplog.text
    assert "checkpoint_bytes=" in caplog.text
    assert "load_elapsed=" in caplog.text


def test_pipeline_growth_stage_skips_no_op_checkpoint_when_limit_already_reached(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An exhausted resumed growth worker should not write a new generation."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    branch_limit = 1_000_000
    checkpoint_path = paths.runtime_checkpoint_path_for_generation(430)
    checkpoint_runner = AnemoneMorpionSearchRunner()
    checkpoint_runner.load_or_create(None, None)
    checkpoint_runner.grow(1)
    checkpoint_runner.save_checkpoint(checkpoint_path)
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=430,
            cycle_index=429,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=branch_limit,
            last_save_unix_s=0.0,
            latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint_path),
        ),
        paths.run_state_path,
    )
    runner = FakeMorpionSearchRunner(
        tree_sizes=(branch_limit,),
        target_values=(1.0,),
        branch_counts=(branch_limit,),
    )

    with caplog.at_level(logging.INFO):
        run_state = run_pipeline_growth_stage(
            replace(_artifact_pipeline_args(tmp_path), tree_branch_limit=branch_limit),
            runner,
            max_cycles=3,
        )

    assert run_state.generation == 430
    assert run_state.cycle_index == 430
    assert run_state.metadata["growth_status"] == "growth_budget_already_exhausted"
    assert run_state.metadata["checkpoint_skipped_reason"] == (
        "no_growth_and_limit_reached"
    )
    assert load_bootstrap_run_state(paths.run_state_path).generation == 430
    assert runner.load_calls == [(str(checkpoint_path), None)]
    assert runner.grow_calls == [5]
    assert runner.checkpoint_calls == []
    assert not paths.runtime_checkpoint_path_for_generation(431).exists()
    assert not paths.tree_snapshot_path_for_generation(431).exists()
    assert not paths.pipeline_manifest_path_for_generation(431).exists()
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert (
        "[growth] no_op_limit_reached branch_count=1000000 limit=1000000 checkpoint_skipped=true"
        in messages
    )
    assert "[save] skipped reason=no_growth_changes nodes_added=0" in messages
    assert "[pipeline] growth_stop reason=growth_budget_already_exhausted" in messages


def test_pipeline_growth_stage_skips_checkpoint_when_time_elapsed_but_no_growth(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Elapsed time alone should not checkpoint an unchanged resumed tree."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=7,
            cycle_index=11,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=100,
            last_save_unix_s=0.0,
        ),
        paths.run_state_path,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(100,), target_values=(1.0,))
    args = replace(
        _artifact_pipeline_args(tmp_path),
        save_after_tree_growth_factor=10.0,
        save_after_seconds=0.0,
    )

    with caplog.at_level(logging.INFO):
        run_state = run_pipeline_growth_stage(args, runner, max_cycles=1)

    assert run_state.generation == 7
    assert run_state.cycle_index == 12
    assert load_bootstrap_run_state(paths.run_state_path).generation == 7
    assert runner.grow_calls == [5]
    assert runner.checkpoint_calls == []
    assert not paths.runtime_checkpoint_path_for_generation(8).exists()
    assert not paths.tree_snapshot_path_for_generation(8).exists()
    assert not paths.pipeline_manifest_path_for_generation(8).exists()
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[save] skipped reason=no_growth_changes nodes_added=0" in messages
    assert "generation_000008" not in messages


def test_pipeline_growth_stage_without_active_model_uses_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Growth should start without an evaluator bundle when no active model exists."""
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))

    with caplog.at_level(logging.INFO):
        run_pipeline_growth_stage(
            _artifact_pipeline_args(tmp_path), runner, max_cycles=1
        )

    assert runner.load_calls == [(None, None)]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert (
        "[growth] active_model_status source=none evaluator=none model_bundle=none"
        in messages
    )


def test_pipeline_growth_stage_uses_pipeline_active_model_for_restore(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Growth should resolve the active model bundle from pipeline/active_model.json."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    model_bundle_path = paths.model_bundle_path_for_generation(8, "linear_5")
    model_bundle_path.mkdir(parents=True, exist_ok=True)
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=8,
            evaluator_name="linear_5",
            model_bundle_path=paths.relative_to_work_dir(model_bundle_path),
            updated_at_utc="2026-04-29T12:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=0,
            cycle_index=0,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths={"stale": "models/generation_000001/stale"},
            active_evaluator_name="stale",
            tree_size_at_last_save=0,
            last_save_unix_s=0.0,
        ),
        paths.run_state_path,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))

    with caplog.at_level(logging.INFO):
        run_pipeline_growth_stage(
            _artifact_pipeline_args(tmp_path), runner, max_cycles=1
        )

    assert runner.load_calls == [(None, str(model_bundle_path))]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert (
        "[growth] active_model_status source=pipeline_active_model generation=8 evaluator=linear_5 "
        f"model_bundle={model_bundle_path}"
    ) in messages


def test_pipeline_growth_stage_missing_active_model_bundle_logs_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Growth should warn and skip attachment when the active bundle path is missing."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    missing_bundle_path = paths.model_bundle_path_for_generation(8, "linear_5")
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=8,
            evaluator_name="linear_5",
            model_bundle_path=paths.relative_to_work_dir(missing_bundle_path),
            updated_at_utc="2026-04-29T12:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))

    with caplog.at_level(logging.INFO):
        run_pipeline_growth_stage(
            _artifact_pipeline_args(tmp_path), runner, max_cycles=1
        )

    assert runner.load_calls == [(None, None)]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert (
        "[growth] active_model_missing_bundle source=pipeline_active_model generation=8 evaluator=linear_5"
        in messages
    )
    assert (
        "[growth] active_model_status source=none evaluator=none model_bundle=none"
        in messages
    )


def test_pipeline_growth_ram_guard_defers_before_tree_growth(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low RAM after restore should defer before patch application and growth."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    available_values = iter((6000.0, 1024.0))

    monkeypatch.setattr(
        pipeline_memory_module,
        "available_ram_mb",
        lambda: next(available_values),
    )

    with caplog.at_level(logging.INFO):
        run_state = run_pipeline_growth_stage(
            replace(_artifact_pipeline_args(tmp_path), min_available_ram_mb=5000),
            runner,
            max_cycles=1,
        )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    persisted_state = load_bootstrap_run_state(paths.run_state_path)

    assert run_state.generation == 0
    assert persisted_state.generation == 0
    assert runner.load_calls == [(None, None)]
    assert runner.grow_calls == []
    assert runner.call_order == []
    assert runner.checkpoint_calls == []
    assert not paths.pipeline_manifest_path_for_generation(1).exists()
    assert (
        "[ram-guard] stage=growth generation=0 action=tree_growth "
        "available_mb=1024.0 required_mb=5000 decision=skip"
    ) in messages
    assert "growth_skip generation=0 reason=low_available_ram action=tree_growth" in messages


def test_pipeline_growth_stage_then_dataset_then_training(tmp_path: Path) -> None:
    """Staged growth, dataset, and training should hand off purely through artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    args = _artifact_pipeline_args(tmp_path)

    run_pipeline_growth_stage(args, runner, max_cycles=1)
    manifest_after_dataset = run_pipeline_dataset_stage(args, generation=1)
    manifest_after_training = run_pipeline_training_stage(args, generation=1)

    assert manifest_after_dataset.dataset_status == "done"
    assert manifest_after_dataset.rows_path == "rows/generation_000001.jsonl"
    assert manifest_after_dataset.training_status == "not_started"
    assert paths.rows_jsonl_path_for_generation(1).is_file()
    assert manifest_after_training.training_status == "done"
    assert manifest_after_training.selected_evaluator_name is not None
    assert paths.pipeline_active_model_path.is_file()


def test_growth_stage_consumes_pending_reevaluation_patch_before_growth(
    tmp_path: Path,
) -> None:
    """Growth stage should apply and delete one pending reevaluation patch before grow."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    patch = _make_reevaluation_patch(patch_id="patch-before-grow")
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)
    runner = FakeMorpionSearchRunner(
        tree_sizes=(5,),
        target_values=(1.0,),
        patch_apply_result=1,
    )

    run_state = run_pipeline_growth_stage(
        _artifact_pipeline_args(tmp_path),
        runner,
        max_cycles=1,
    )
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))

    assert runner.call_order == [f"apply_patch:{patch.patch_id}", "grow"]
    assert runner.received_patches == [patch]
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert run_state.generation == 1
    assert manifest.tree_snapshot_path == "tree_exports/generation_000001.json"
    assert manifest.runtime_checkpoint_path == paths.relative_to_work_dir(
        paths.runtime_checkpoint_path_for_generation(1)
    )


def test_growth_stage_keeps_patch_when_patch_application_fails(tmp_path: Path) -> None:
    """Growth stage should leave a pending patch in place when apply fails."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    patch = _make_reevaluation_patch(patch_id="patch-fail")
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)
    runner = FakeMorpionSearchRunner(
        tree_sizes=(5,),
        target_values=(1.0,),
        patch_apply_error=RuntimeError("patch failed"),
    )

    with pytest.raises(RuntimeError, match="patch failed"):
        run_pipeline_growth_stage(
            _artifact_pipeline_args(tmp_path),
            runner,
            max_cycles=1,
        )

    assert runner.call_order == [f"apply_patch:{patch.patch_id}"]
    assert paths.pipeline_reevaluation_patch_path.exists()
    assert runner.grow_calls == []


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


def test_dataset_stage_extracts_rows_from_manifest_tree_snapshot(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset stage should extract rows and mark the manifest done."""
    leaderboard_calls: list[tuple[int, int]] = []

    def _persist_leaderboard(**kwargs: object) -> None:
        leaderboard_calls.append(
            (int(kwargs["generation"]), int(kwargs["cycle_index"]))
        )

    monkeypatch.setattr(
        pipeline_stages_module,
        "persist_certified_leaderboard_candidates",
        _persist_leaderboard,
    )

    def _unexpected_full_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError

    monkeypatch.setattr(
        pipeline_stages_module,
        "save_morpion_supervised_rows",
        _unexpected_full_save,
    )
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

    with caplog.at_level(logging.INFO):
        manifest = run_pipeline_dataset_stage(
            _artifact_pipeline_args(tmp_path), generation=1
        )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    dataset_status = load_pipeline_dataset_status_file(
        paths.pipeline_dataset_status_path_for_generation(1)
    )
    loaded_rows = load_morpion_supervised_rows(paths.rows_jsonl_path_for_generation(1))

    assert manifest.rows_path == "rows/generation_000001.jsonl"
    assert paths.rows_jsonl_path_for_generation(1).is_file()
    assert manifest.dataset_status == "done"
    assert manifest.training_status == "not_started"
    assert not paths.pipeline_dataset_claim_path_for_generation(1).exists()
    assert manifest.metadata["dataset_rows"] == 1
    assert manifest.metadata["dataset_rows"] == len(loaded_rows.rows)
    assert loaded_rows.metadata["bootstrap_generation"] == 1
    assert dataset_status.record_status is not None
    assert dataset_status.record_status.current_best_total_points == 37
    assert dataset_status.frontier_status is not None
    assert dataset_status.frontier_status.current_best_total_points == 38
    assert leaderboard_calls == [(1, 1)]
    assert "[pipeline] dataset_claim_created generation=1" in messages
    assert "[pipeline] dataset_export_start generation=1" in messages
    assert "[pipeline] dataset_rows_stream_done generation=1 rows=1" in messages
    assert "[pipeline] dataset_export_done generation=1 rows=1" in messages
    assert "[pipeline] dataset_manifest_written generation=1" in messages
    assert "[pipeline-memory] stage=dataset generation=1 event=after_snapshot_load" in messages
    assert "[pipeline-memory] stage=dataset generation=1 event=after_rows_write_stream" in messages


def test_dataset_stage_rejects_streamed_rows_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset stage should not mark done when metadata row count drifts."""
    monkeypatch.setattr(
        pipeline_stages_module,
        "persist_certified_leaderboard_candidates",
        lambda **kwargs: None,
    )
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
    original_streaming_rows = pipeline_stages_module._streaming_rows_from_training_snapshot

    def _streaming_rows_with_bad_metadata(**kwargs: object) -> object:
        streaming_rows = original_streaming_rows(**kwargs)
        return replace(
            streaming_rows,
            metadata={**streaming_rows.metadata, "num_rows": 999},
        )

    monkeypatch.setattr(
        pipeline_stages_module,
        "_streaming_rows_from_training_snapshot",
        _streaming_rows_with_bad_metadata,
    )

    with pytest.raises(RuntimeError, match="metadata count mismatch"):
        run_pipeline_dataset_stage(_artifact_pipeline_args(tmp_path), generation=1)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert manifest.dataset_status == "failed"


def test_dataset_stage_ram_guard_defers_before_snapshot_load(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low available RAM should defer dataset extraction without failing it."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    snapshot_path = paths.tree_snapshot_path_for_generation(1)
    save_training_tree_snapshot(
        _make_training_snapshot(target_value=1.25, root_node_id="node-0"),
        snapshot_path,
    )
    original_manifest = MorpionPipelineGenerationManifest(
        generation=1,
        created_at_utc="2026-04-28T12:00:00Z",
        tree_snapshot_path=paths.relative_to_work_dir(snapshot_path),
        dataset_status="not_started",
        training_status="not_started",
    )
    save_pipeline_manifest(original_manifest, paths.pipeline_manifest_path_for_generation(1))

    def _unexpected_snapshot_load(**kwargs: object) -> TrainingTreeSnapshot:
        del kwargs
        raise AssertionError

    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: 1024.0)
    monkeypatch.setattr(
        pipeline_stages_module,
        "_load_training_snapshot_for_generation",
        _unexpected_snapshot_load,
    )

    with caplog.at_level(logging.INFO):
        returned_manifest = run_pipeline_dataset_stage(
            replace(_artifact_pipeline_args(tmp_path), min_available_ram_mb=5000),
            generation=1,
        )

    persisted_manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    messages = "\n".join(record.getMessage() for record in caplog.records)

    assert returned_manifest == original_manifest
    assert persisted_manifest.dataset_status == "not_started"
    assert not paths.rows_path_for_generation(1).exists()
    assert not paths.rows_jsonl_path_for_generation(1).exists()
    assert not paths.pipeline_dataset_claim_path_for_generation(1).exists()
    assert (
        "[ram-guard] stage=dataset generation=1 action=snapshot_load "
        "available_mb=1024.0 required_mb=5000 decision=skip"
    ) in messages
    assert "dataset_skip generation=1 reason=low_available_ram" in messages


def test_pipeline_sharded_export_and_dataset_stage_round_trip(tmp_path: Path) -> None:
    """Artifact-pipeline stages should round-trip through sharded tree exports."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    args = replace(_artifact_pipeline_args(tmp_path), training_export_mode="sharded")

    run_pipeline_growth_stage(args, runner, max_cycles=1)
    manifest_after_growth = load_pipeline_manifest(
        paths.pipeline_manifest_path_for_generation(1)
    )
    manifest_after_dataset = run_pipeline_dataset_stage(args, generation=1)

    assert manifest_after_growth.tree_snapshot_path == (
        "tree_exports_sharded/generation_000001.json"
    )
    assert paths.sharded_tree_snapshot_path_for_generation(1).is_file()
    assert manifest_after_dataset.dataset_status == "done"
    assert manifest_after_dataset.rows_path == "rows/generation_000001.jsonl"
    assert paths.rows_jsonl_path_for_generation(1).is_file()


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


def test_dataset_stage_marks_failed_on_exception(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(
            FileNotFoundError,
            match=r"Pipeline tree snapshot does not exist: .*generation_000001.json",
        ),
    ):
        run_pipeline_dataset_stage(_artifact_pipeline_args(tmp_path), generation=1)

    messages = "\n".join(record.getMessage() for record in caplog.records)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert manifest.dataset_status == "failed"
    assert not paths.pipeline_dataset_claim_path_for_generation(1).exists()
    assert "dataset_skip generation=1 reason=missing_tree_export" in messages


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

    manifest = run_pipeline_training_stage(
        _artifact_pipeline_args(tmp_path), generation=1
    )
    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)
    training_status = load_pipeline_training_status_file(
        paths.pipeline_training_status_path_for_generation(1)
    )

    assert manifest.training_status == "done"
    assert manifest.selected_evaluator_name is not None
    assert paths.pipeline_active_model_path.is_file()
    assert active_model.evaluator_name == manifest.selected_evaluator_name
    assert (
        active_model.model_bundle_path
        == manifest.model_bundle_paths[manifest.selected_evaluator_name]
    )
    assert training_status.selected_evaluator_name == manifest.selected_evaluator_name
    assert training_status.selection_policy == "lowest_final_loss"
    assert set(training_status.evaluator_results) == set(manifest.model_bundle_paths)
    assert isinstance(
        training_status.evaluator_results[manifest.selected_evaluator_name],
        MorpionPipelineEvaluatorTrainingResult,
    )
    assert (
        training_status.evaluator_results[
            manifest.selected_evaluator_name
        ].model_bundle_path
        == manifest.model_bundle_paths[manifest.selected_evaluator_name]
    )
    assert not paths.pipeline_training_claim_path_for_generation(1).exists()


def test_training_stage_default_trains_all_configured_evaluators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a subset request, training should receive the full evaluator family."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _prepare_training_stage_input(
        paths,
        generation=1,
        rows=_make_rows_with_count(3),
    )
    args = replace(
        _artifact_pipeline_args(tmp_path),
        evaluators_config=_multi_evaluator_config(),
    )
    trained_names: list[tuple[str, ...]] = []
    trained_row_counts: list[int] = []
    training_rows_paths: list[Path] = []

    def _fake_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        resolved_config = cast(
            "MorpionEvaluatorsConfig",
            kwargs["resolved_evaluators_config"],
        )
        evaluator_names = tuple(resolved_config.evaluators)
        trained_names.append(evaluator_names)
        trained_rows = cast("MorpionSupervisedRows", kwargs["rows"])
        trained_row_counts.append(len(trained_rows.rows))
        training_rows_paths.append(cast("Path", kwargs["rows_path"]))
        return _fake_training_result_for_evaluators(
            paths,
            generation=1,
            evaluator_names=evaluator_names,
            selected_evaluator_name="linear_5",
        )

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _fake_train_and_select,
    )

    manifest = run_pipeline_training_stage(args, generation=1)

    assert trained_names == [("linear_5", "mlp_5")]
    assert trained_row_counts == [3]
    assert training_rows_paths == [paths.rows_path_for_generation(1)]
    assert set(manifest.model_bundle_paths) == {"linear_5", "mlp_5"}
    assert manifest.metadata["training_evaluator_names"] == ["linear_5", "mlp_5"]
    assert "training_max_rows" not in manifest.metadata
    assert "skip_evaluator_diagnostics" not in manifest.metadata
    assert manifest.metadata["evaluator_diagnostics_max_rows"] == 60
    assert manifest.metadata["evaluator_diagnostics_sample_policy"] == "first_n"


def test_training_stage_restricts_to_one_requested_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single requested evaluator should be the only trained and persisted model."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _prepare_training_stage_input(paths, generation=1)
    args = replace(
        _artifact_pipeline_args(tmp_path),
        evaluators_config=_multi_evaluator_config(),
        training_evaluator_names=("linear_5",),
    )
    trained_names: list[tuple[str, ...]] = []

    def _fake_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        resolved_config = cast(
            "MorpionEvaluatorsConfig",
            kwargs["resolved_evaluators_config"],
        )
        evaluator_names = tuple(resolved_config.evaluators)
        trained_names.append(evaluator_names)
        return _fake_training_result_for_evaluators(
            paths,
            generation=1,
            evaluator_names=evaluator_names,
            selected_evaluator_name="linear_5",
        )

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _fake_train_and_select,
    )

    manifest = run_pipeline_training_stage(args, generation=1)
    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)

    assert trained_names == [("linear_5",)]
    assert set(manifest.model_bundle_paths) == {"linear_5"}
    assert manifest.selected_evaluator_name == "linear_5"
    assert manifest.metadata["training_evaluator_names"] == ["linear_5"]
    assert active_model.evaluator_name == "linear_5"


def test_training_stage_restricts_to_requested_evaluator_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested evaluator pair should preserve requested order and selection."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _prepare_training_stage_input(paths, generation=1)
    args = replace(
        _artifact_pipeline_args(tmp_path),
        evaluators_config=_multi_evaluator_config(),
        training_evaluator_names=("mlp_5", "linear_5"),
    )
    trained_names: list[tuple[str, ...]] = []

    def _fake_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        resolved_config = cast(
            "MorpionEvaluatorsConfig",
            kwargs["resolved_evaluators_config"],
        )
        evaluator_names = tuple(resolved_config.evaluators)
        trained_names.append(evaluator_names)
        return _fake_training_result_for_evaluators(
            paths,
            generation=1,
            evaluator_names=evaluator_names,
            selected_evaluator_name="mlp_5",
        )

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _fake_train_and_select,
    )

    manifest = run_pipeline_training_stage(args, generation=1)

    assert trained_names == [("mlp_5", "linear_5")]
    assert set(manifest.model_bundle_paths) == {"linear_5", "mlp_5"}
    assert manifest.selected_evaluator_name == "mlp_5"
    assert manifest.metadata["training_evaluator_names"] == ["mlp_5", "linear_5"]


def test_training_stage_debug_controls_limit_rows_and_record_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training debug controls should combine with evaluator subset selection."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _prepare_training_stage_input(
        paths,
        generation=1,
        rows=_make_rows_with_count(5),
    )
    args = replace(
        _artifact_pipeline_args(tmp_path),
        evaluators_config=_multi_evaluator_config(),
        training_evaluator_names=("linear_5",),
        training_max_rows=2,
        skip_evaluator_diagnostics=True,
    )
    trained_names: list[tuple[str, ...]] = []
    trained_row_counts: list[int] = []
    training_rows_paths: list[Path] = []

    def _fake_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        resolved_config = cast(
            "MorpionEvaluatorsConfig",
            kwargs["resolved_evaluators_config"],
        )
        evaluator_names = tuple(resolved_config.evaluators)
        trained_names.append(evaluator_names)
        trained_rows = cast("MorpionSupervisedRows", kwargs["rows"])
        trained_row_counts.append(len(trained_rows.rows))
        training_rows_paths.append(cast("Path", kwargs["rows_path"]))
        return _fake_training_result_for_evaluators(
            paths,
            generation=1,
            evaluator_names=evaluator_names,
            selected_evaluator_name="linear_5",
        )

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _fake_train_and_select,
    )

    manifest = run_pipeline_training_stage(args, generation=1)
    persisted_subset = load_morpion_supervised_rows(training_rows_paths[0])

    assert trained_names == [("linear_5",)]
    assert trained_row_counts == [2]
    assert training_rows_paths == [
        paths.rows_dir / "generation_000001.training_subset.json"
    ]
    assert len(persisted_subset.rows) == 2
    assert [row.node_id for row in persisted_subset.rows] == ["row-1", "row-2"]
    assert set(manifest.model_bundle_paths) == {"linear_5"}
    assert manifest.metadata["training_evaluator_names"] == ["linear_5"]
    assert manifest.metadata["training_max_rows"] == 2
    assert manifest.metadata["training_rows_used"] == 2
    assert manifest.metadata["training_rows_original"] == 5
    assert manifest.metadata["skip_evaluator_diagnostics"] is True


def test_training_stage_streams_jsonl_rows_without_materialized_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSONL training should use the streaming path and avoid subset artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _prepare_training_stage_jsonl_input(
        paths,
        generation=1,
        rows=_make_rows_with_count(5),
    )
    args = replace(
        _artifact_pipeline_args(tmp_path),
        evaluators_config=_multi_evaluator_config(),
        training_evaluator_names=("linear_5",),
        training_max_rows=2,
        training_row_chunk_size=2,
        skip_evaluator_diagnostics=True,
    )
    streaming_calls: list[tuple[int | None, int, str, int | None]] = []

    def _unexpected_full_load(path: str | Path) -> MorpionSupervisedRows:
        del path
        raise AssertionError

    def _unexpected_materialized_train(**kwargs: object) -> BootstrapTrainingResult:
        del kwargs
        raise AssertionError

    def _fake_streaming_train(**kwargs: object) -> BootstrapTrainingResult:
        rows_source = cast("Any", kwargs["rows_source"])
        streaming_calls.append(
            (
                cast("int | None", kwargs["max_rows"]),
                int(kwargs["chunk_size"]),
                str(rows_source.format_kind),
                cast("int | None", rows_source.row_count),
            )
        )
        return _fake_training_result_for_evaluators(
            paths,
            generation=1,
            evaluator_names=("linear_5",),
            selected_evaluator_name="linear_5",
        )

    monkeypatch.setattr(
        pipeline_stages_module,
        "load_morpion_supervised_rows",
        _unexpected_full_load,
    )
    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _unexpected_materialized_train,
    )
    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators_streaming",
        _fake_streaming_train,
    )

    manifest = run_pipeline_training_stage(args, generation=1)

    assert streaming_calls == [(2, 2, "jsonl", 5)]
    assert not (paths.rows_dir / "generation_000001.training_subset.json").exists()
    assert manifest.metadata["training_row_source_format"] == "jsonl"
    assert manifest.metadata["training_row_chunk_size"] == 2
    assert manifest.metadata["training_split_policy"] == "index_modulo_5"
    assert manifest.metadata["training_rows_original"] == 5
    assert manifest.metadata["training_rows_used"] == 2
    assert manifest.metadata["training_max_rows"] == 2
    assert manifest.metadata["skip_evaluator_diagnostics"] is True


def test_training_stage_unknown_training_evaluator_name_raises_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown subset names should fail before evaluator training begins."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _prepare_training_stage_input(paths, generation=1)
    args = replace(
        _artifact_pipeline_args(tmp_path),
        evaluators_config=_multi_evaluator_config(),
        training_evaluator_names=("linear_5", "missing"),
    )

    def _unexpected_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        del kwargs
        raise AssertionError

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _unexpected_train_and_select,
    )

    with pytest.raises(
        ValueError,
        match="Unknown requested training evaluator names: missing",
    ):
        run_pipeline_training_stage(args, generation=1)

    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    assert manifest.training_status == "failed"
    assert cursor.latest_started_generation is None
    assert cursor.latest_completed_generation is None


def test_train_and_select_evaluators_runs_diagnostics_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default training should still persist evaluator diagnostics."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows = _make_rows_with_count(3)
    rows_path = paths.rows_path_for_generation(1)
    save_morpion_supervised_rows(rows, rows_path)
    resolved_config = MorpionEvaluatorsConfig(
        evaluators={"linear_5": _multi_evaluator_config().evaluators["linear_5"]}
    )
    diagnostics_calls: list[str] = []

    def _fake_train_morpion_regressor(training_args: object) -> tuple[object, dict[str, object]]:
        del training_args
        return object(), {
            "final_loss": 0.25,
            "train_loss": 0.25,
            "validation_loss": None,
            "num_epochs": 1,
            "num_samples": len(rows.rows),
            "batch_size": 1,
            "learning_rate": 1e-3,
        }

    def _fake_persist_diagnostics(**kwargs: object) -> None:
        diagnostics_calls.append(cast("str", kwargs["evaluator_name"]))

    monkeypatch.setattr(
        cycle_training_module,
        "train_morpion_regressor",
        _fake_train_morpion_regressor,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "persist_evaluator_training_diagnostics",
        _fake_persist_diagnostics,
    )

    memory = MemoryDiagnostics(MemoryDiagnosticsConfig(enabled=False))
    try:
        training_result = train_and_select_evaluators(
            args=_artifact_pipeline_args(tmp_path),
            paths=paths,
            run_state=initialize_bootstrap_run_state(),
            rows=rows,
            rows_path=rows_path,
            generation=1,
            timestamp_utc="2026-04-28T12:00:00Z",
            resolved_evaluators_config=resolved_config,
            resolved_control=MorpionBootstrapControl(),
            memory=memory,
        )
    finally:
        memory.close()

    assert diagnostics_calls == ["linear_5"]
    assert training_result.selected_evaluator_name == "linear_5"


def test_train_and_select_evaluators_bounds_materialized_diagnostics_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Materialized-row diagnostics should use a first-N sample when bounded."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows = _make_rows_with_count(5)
    rows_path = paths.rows_path_for_generation(1)
    save_morpion_supervised_rows(rows, rows_path)
    resolved_config = MorpionEvaluatorsConfig(
        evaluators={"linear_5": _multi_evaluator_config().evaluators["linear_5"]}
    )
    diagnostics_rows: list[object] = []

    def _fake_train_morpion_regressor(training_args: object) -> tuple[object, dict[str, object]]:
        del training_args
        return object(), {
            "final_loss": 0.25,
            "train_loss": 0.25,
            "validation_loss": None,
            "num_epochs": 1,
            "num_samples": len(rows.rows),
            "batch_size": 1,
            "learning_rate": 1e-3,
        }

    def _fake_persist_diagnostics(**kwargs: object) -> None:
        diagnostics_rows.append(kwargs["rows"])

    monkeypatch.setattr(
        cycle_training_module,
        "train_morpion_regressor",
        _fake_train_morpion_regressor,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "persist_evaluator_training_diagnostics",
        _fake_persist_diagnostics,
    )

    memory = MemoryDiagnostics(MemoryDiagnosticsConfig(enabled=False))
    try:
        with caplog.at_level(logging.INFO):
            train_and_select_evaluators(
                args=replace(
                    _artifact_pipeline_args(tmp_path),
                    evaluator_diagnostics_max_rows=2,
                ),
                paths=paths,
                run_state=initialize_bootstrap_run_state(),
                rows=rows,
                rows_path=rows_path,
                generation=1,
                timestamp_utc="2026-04-28T12:00:00Z",
                resolved_evaluators_config=resolved_config,
                resolved_control=MorpionBootstrapControl(),
                memory=memory,
            )
    finally:
        memory.close()

    diagnostic_rows = cast("Any", diagnostics_rows[0])
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert len(diagnostic_rows.rows) == 2
    assert diagnostic_rows.metadata["diagnostic_sample_policy"] == "first_n"
    assert diagnostic_rows.metadata["diagnostic_sample_rows"] == 2
    assert diagnostic_rows.metadata["diagnostic_sample_max_rows"] == 2
    assert diagnostic_rows.metadata["diagnostic_source_format"] == "json"
    assert (
        "[diagnostics] sampled generation=1 evaluator=linear_5 rows=2 "
        "max_rows=2 policy=first_n source_format=json"
    ) in messages


def test_train_and_select_evaluators_keeps_full_materialized_diagnostics_when_unbounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialized JSON diagnostics can still use all rows when explicitly unbounded."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows = _make_rows_with_count(5)
    rows_path = paths.rows_path_for_generation(1)
    save_morpion_supervised_rows(rows, rows_path)
    resolved_config = MorpionEvaluatorsConfig(
        evaluators={"linear_5": _multi_evaluator_config().evaluators["linear_5"]}
    )
    diagnostics_rows: list[object] = []

    def _fake_train_morpion_regressor(training_args: object) -> tuple[object, dict[str, object]]:
        del training_args
        return object(), {
            "final_loss": 0.25,
            "train_loss": 0.25,
            "validation_loss": None,
            "num_epochs": 1,
            "num_samples": len(rows.rows),
            "batch_size": 1,
            "learning_rate": 1e-3,
        }

    def _fake_persist_diagnostics(**kwargs: object) -> None:
        diagnostics_rows.append(kwargs["rows"])

    monkeypatch.setattr(
        cycle_training_module,
        "train_morpion_regressor",
        _fake_train_morpion_regressor,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "persist_evaluator_training_diagnostics",
        _fake_persist_diagnostics,
    )

    memory = MemoryDiagnostics(MemoryDiagnosticsConfig(enabled=False))
    try:
        train_and_select_evaluators(
            args=replace(
                _artifact_pipeline_args(tmp_path),
                evaluator_diagnostics_max_rows=None,
            ),
            paths=paths,
            run_state=initialize_bootstrap_run_state(),
            rows=rows,
            rows_path=rows_path,
            generation=1,
            timestamp_utc="2026-04-28T12:00:00Z",
            resolved_evaluators_config=resolved_config,
            resolved_control=MorpionBootstrapControl(),
            memory=memory,
        )
    finally:
        memory.close()

    diagnostic_rows = cast("Any", diagnostics_rows[0])
    assert len(diagnostic_rows.rows) == 5
    assert "diagnostic_sample_policy" not in diagnostic_rows.metadata


def test_train_and_select_evaluators_streaming_bounds_jsonl_diagnostics_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Streaming JSONL diagnostics should sample rows without materializing the artifact."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows = _make_rows_with_count(5)
    rows_path = paths.rows_jsonl_path_for_generation(1)
    save_morpion_supervised_rows_streaming(
        rows=rows.rows,
        metadata={**rows.metadata, "num_rows": len(rows.rows)},
        path=rows_path,
    )
    rows_source = morpion_supervised_rows_source_from_path(rows_path)
    resolved_config = MorpionEvaluatorsConfig(
        evaluators={"linear_5": _multi_evaluator_config().evaluators["linear_5"]}
    )
    diagnostics_rows: list[object] = []

    def _unexpected_full_load(path: str | Path) -> MorpionSupervisedRows:
        del path
        raise AssertionError

    def _fake_train_morpion_regressor_streaming(
        training_args: object,
    ) -> tuple[object, dict[str, object]]:
        del training_args
        return object(), {
            "final_loss": 0.25,
            "train_loss": 0.25,
            "validation_loss": None,
            "num_epochs": 1,
            "num_samples": len(rows.rows),
            "batch_size": 1,
            "learning_rate": 1e-3,
        }

    def _fake_persist_diagnostics(**kwargs: object) -> None:
        diagnostics_rows.append(kwargs["rows"])

    monkeypatch.setattr(
        tree_to_dataset_module,
        "load_morpion_supervised_rows",
        _unexpected_full_load,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "load_previous_evaluator_for_diagnostics",
        lambda path: None,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "train_morpion_regressor_streaming",
        _fake_train_morpion_regressor_streaming,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "persist_evaluator_training_diagnostics",
        _fake_persist_diagnostics,
    )

    memory = MemoryDiagnostics(MemoryDiagnosticsConfig(enabled=False))
    try:
        with caplog.at_level(logging.INFO):
            cycle_training_module.train_and_select_evaluators_streaming(
                args=replace(
                    _artifact_pipeline_args(tmp_path),
                    evaluator_diagnostics_max_rows=2,
                ),
                paths=paths,
                run_state=initialize_bootstrap_run_state(),
                rows_path=rows_path,
                rows_source=rows_source,
                generation=1,
                timestamp_utc="2026-04-28T12:00:00Z",
                resolved_evaluators_config=resolved_config,
                resolved_control=MorpionBootstrapControl(),
                memory=memory,
                max_rows=None,
                chunk_size=2,
            )
    finally:
        memory.close()

    diagnostic_rows = cast("Any", diagnostics_rows[0])
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert len(diagnostic_rows.rows) == 2
    assert diagnostic_rows.metadata["diagnostic_sample_policy"] == "first_n"
    assert diagnostic_rows.metadata["diagnostic_sample_rows"] == 2
    assert diagnostic_rows.metadata["diagnostic_sample_max_rows"] == 2
    assert diagnostic_rows.metadata["diagnostic_source_format"] == "jsonl"
    assert (
        "[diagnostics] sampled generation=1 evaluator=linear_5 rows=2 "
        "max_rows=2 policy=first_n source_format=jsonl"
    ) in messages


def test_train_and_select_evaluators_can_skip_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skip diagnostics should avoid the expensive diagnostics persistence path."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows = _make_rows_with_count(3)
    rows_path = paths.rows_path_for_generation(1)
    save_morpion_supervised_rows(rows, rows_path)
    resolved_config = MorpionEvaluatorsConfig(
        evaluators={"linear_5": _multi_evaluator_config().evaluators["linear_5"]}
    )

    def _fake_train_morpion_regressor(training_args: object) -> tuple[object, dict[str, object]]:
        del training_args
        return object(), {
            "final_loss": 0.25,
            "train_loss": 0.25,
            "validation_loss": None,
            "num_epochs": 1,
            "num_samples": len(rows.rows),
            "batch_size": 1,
            "learning_rate": 1e-3,
        }

    def _unexpected_persist_diagnostics(**kwargs: object) -> None:
        del kwargs
        raise AssertionError

    def _unexpected_previous_model_load(path: object) -> object:
        del path
        raise AssertionError

    monkeypatch.setattr(
        cycle_training_module,
        "train_morpion_regressor",
        _fake_train_morpion_regressor,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "persist_evaluator_training_diagnostics",
        _unexpected_persist_diagnostics,
    )
    monkeypatch.setattr(
        cycle_training_module,
        "load_previous_evaluator_for_diagnostics",
        _unexpected_previous_model_load,
    )

    memory = MemoryDiagnostics(MemoryDiagnosticsConfig(enabled=False))
    try:
        with caplog.at_level(logging.INFO):
            training_result = train_and_select_evaluators(
                args=replace(
                    _artifact_pipeline_args(tmp_path),
                    skip_evaluator_diagnostics=True,
                ),
                paths=paths,
                run_state=initialize_bootstrap_run_state(),
                rows=rows,
                rows_path=rows_path,
                generation=1,
                timestamp_utc="2026-04-28T12:00:00Z",
                resolved_evaluators_config=resolved_config,
                resolved_control=MorpionBootstrapControl(),
                memory=memory,
            )
    finally:
        memory.close()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert training_result.selected_evaluator_name == "linear_5"
    assert (
        "[diagnostics] skipped generation=1 evaluator=linear_5 "
        "reason=skip_evaluator_diagnostics"
    ) in messages


def test_training_stage_logs_active_model_update(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Training stage logs should expose the published active-model generation."""
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

    with caplog.at_level(logging.INFO):
        run_pipeline_training_stage(_artifact_pipeline_args(tmp_path), generation=1)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[pipeline] active_model_update generation=1 evaluator=" in messages


def test_training_stage_writes_started_cursor_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training start should persist the monotonic cursor before long training."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows_path = paths.rows_path_for_generation(6)
    save_morpion_supervised_rows(_make_rows(), rows_path)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=6,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(6),
    )

    def _fake_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        del kwargs
        cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
        assert cursor.latest_started_generation == 6
        assert cursor.latest_completed_generation is None
        return _fake_training_result(paths, generation=6)

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _fake_train_and_select,
    )

    run_pipeline_training_stage(_artifact_pipeline_args(tmp_path), generation=6)
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)

    assert cursor.latest_started_generation == 6
    assert cursor.latest_completed_generation == 6


def test_training_cursor_malformed_json_raises_artifact_error(
    tmp_path: Path,
) -> None:
    """Malformed training cursor artifacts should not be ignored silently."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.pipeline_training_cursor_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pipeline_training_cursor_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(InvalidMorpionPipelineArtifactError):
        load_pipeline_training_cursor(paths.pipeline_training_cursor_path)


def test_training_stage_skips_explicit_stale_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit training should skip stale generations without marking failure."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows_path = paths.rows_path_for_generation(5)
    save_morpion_supervised_rows(_make_rows(), rows_path)
    save_pipeline_training_cursor(
        MorpionPipelineTrainingCursor(latest_started_generation=6),
        paths.pipeline_training_cursor_path,
    )
    original_manifest = MorpionPipelineGenerationManifest(
        generation=5,
        created_at_utc="2026-04-28T12:00:00Z",
        rows_path=paths.relative_to_work_dir(rows_path),
        dataset_status="done",
        training_status="not_started",
    )
    save_pipeline_manifest(original_manifest, paths.pipeline_manifest_path_for_generation(5))

    def _unexpected_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        del kwargs
        raise AssertionError

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _unexpected_train_and_select,
    )

    with caplog.at_level(logging.INFO):
        returned_manifest = run_pipeline_training_stage(
            _artifact_pipeline_args(tmp_path),
            generation=5,
        )

    persisted_manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(5))
    messages = "\n".join(record.getMessage() for record in caplog.records)

    assert returned_manifest == original_manifest
    assert persisted_manifest.training_status == "not_started"
    assert not paths.pipeline_active_model_path.exists()
    assert not paths.pipeline_training_claim_path_for_generation(5).exists()
    assert "training_skip generation=5 reason=stale_generation" in messages


def test_training_stage_ram_guard_defers_before_rows_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Low available RAM should defer training without marking it failed."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _prepare_training_stage_input(paths, generation=5)
    original_manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(5))

    def _unexpected_rows_load(path: str | Path) -> MorpionSupervisedRows:
        del path
        raise AssertionError

    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: 1024.0)
    monkeypatch.setattr(
        pipeline_stages_module,
        "load_morpion_supervised_rows",
        _unexpected_rows_load,
    )

    with caplog.at_level(logging.INFO):
        returned_manifest = run_pipeline_training_stage(
            replace(_artifact_pipeline_args(tmp_path), min_available_ram_mb=5000),
            generation=5,
        )

    persisted_manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(5))
    messages = "\n".join(record.getMessage() for record in caplog.records)

    assert returned_manifest == original_manifest
    assert persisted_manifest.training_status == "not_started"
    assert not paths.pipeline_active_model_path.exists()
    assert not paths.pipeline_training_cursor_path.exists()
    assert not paths.pipeline_training_claim_path_for_generation(5).exists()
    assert (
        "[ram-guard] stage=training generation=5 action=rows_load "
        "available_mb=1024.0 required_mb=5000 decision=skip"
    ) in messages
    assert "training_skip generation=5 reason=low_available_ram" in messages


def test_training_stage_active_model_commit_cannot_go_backward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A late older training result must not overwrite a newer active model."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows_path = paths.rows_path_for_generation(5)
    save_morpion_supervised_rows(_make_rows(), rows_path)
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=4,
            evaluator_name="linear_5",
            model_bundle_path="models/generation_000004/linear_5",
            updated_at_utc="2026-04-28T12:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=5,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(5),
    )

    def _fake_train_and_select(**kwargs: object) -> BootstrapTrainingResult:
        del kwargs
        save_pipeline_active_model(
            MorpionPipelineActiveModel(
                generation=6,
                evaluator_name="mlp_5",
                model_bundle_path="models/generation_000006/mlp_5",
                updated_at_utc="2026-04-28T12:10:00Z",
            ),
            paths.pipeline_active_model_path,
        )
        return _fake_training_result(paths, generation=5)

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        _fake_train_and_select,
    )

    with caplog.at_level(logging.INFO):
        manifest = run_pipeline_training_stage(
            _artifact_pipeline_args(tmp_path),
            generation=5,
        )
    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    messages = "\n".join(record.getMessage() for record in caplog.records)

    assert manifest.training_status == "done"
    assert active_model.generation == 6
    assert active_model.evaluator_name == "mlp_5"
    assert cursor.latest_completed_generation == 5
    assert "active_model_update_skipped generation=5 reason=stale_generation" in messages


def test_training_stage_newer_generation_commits_and_updates_completed_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-stale training result should publish active_model and complete cursor."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    rows_path = paths.rows_path_for_generation(6)
    save_morpion_supervised_rows(_make_rows(), rows_path)
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=5,
            evaluator_name="linear_5",
            model_bundle_path="models/generation_000005/linear_5",
            updated_at_utc="2026-04-28T12:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=6,
            created_at_utc="2026-04-28T12:00:00Z",
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(6),
    )

    monkeypatch.setattr(
        pipeline_stages_module,
        "_train_and_select_evaluators",
        lambda **kwargs: _fake_training_result(paths, generation=6),
    )

    run_pipeline_training_stage(_artifact_pipeline_args(tmp_path), generation=6)
    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)

    assert active_model.generation == 6
    assert active_model.evaluator_name == "linear_5"
    assert cursor.latest_started_generation == 6
    assert cursor.latest_completed_generation == 6


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

    monkeypatch.setattr(
        launcher_module, "run_pipeline_dataset_stage", _fake_dataset_stage
    )

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

    monkeypatch.setattr(
        launcher_module, "run_pipeline_growth_stage", _fake_growth_stage
    )
    monkeypatch.setattr(
        launcher_module, "run_morpion_bootstrap_loop", _unexpected_full_loop
    )

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


def test_artifact_pipeline_reevaluation_stage_dispatches_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reevaluation CLI dispatch should call the one-shot worker."""
    captured: list[tuple[MorpionBootstrapArgs, int]] = []
    fake_result = _make_reevaluation_worker_result()

    def _fake_worker(
        args: MorpionBootstrapArgs,
        *,
        max_nodes_per_patch: int = 10_000,
    ) -> MorpionReevaluationWorkerResult:
        captured.append((args, max_nodes_per_patch))
        return fake_result

    monkeypatch.setattr(
        launcher_module,
        "run_morpion_reevaluation_worker_once",
        _fake_worker,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "reevaluation",
            "--reevaluation-max-nodes-per-patch",
            "123",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )
    result = run_morpion_bootstrap_experiment(launcher_args)

    assert result is fake_result
    assert len(captured) == 1
    assert captured[0][0].pipeline_mode == "artifact_pipeline"
    assert captured[0][1] == 123


def test_artifact_pipeline_reevaluation_stage_uses_default_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reevaluation CLI dispatch should default to 10000 nodes per patch."""
    captured: list[int] = []
    fake_result = _make_reevaluation_worker_result()

    def _fake_worker(
        args: MorpionBootstrapArgs,
        *,
        max_nodes_per_patch: int = 10_000,
    ) -> MorpionReevaluationWorkerResult:
        del args
        captured.append(max_nodes_per_patch)
        return fake_result

    monkeypatch.setattr(
        launcher_module,
        "run_morpion_reevaluation_worker_once",
        _fake_worker,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "reevaluation",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )
    result = run_morpion_bootstrap_experiment(launcher_args)

    assert result is fake_result
    assert captured == [10_000]


def test_artifact_pipeline_reevaluation_stage_does_not_build_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reevaluation dispatch should not construct the Anemone search runner."""
    fake_result = _make_reevaluation_worker_result()

    def _unexpected_runner(*args: object, **kwargs: object) -> object:
        raise _unexpected_reevaluation_runner_error()

    monkeypatch.setattr(
        launcher_module,
        "AnemoneMorpionSearchRunner",
        _unexpected_runner,
    )
    monkeypatch.setattr(
        launcher_module,
        "run_morpion_reevaluation_worker_once",
        lambda args, *, max_nodes_per_patch=10_000: fake_result,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "reevaluation",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    assert run_morpion_bootstrap_experiment(launcher_args) is fake_result


def test_artifact_pipeline_reevaluation_negative_batch_size_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative reevaluation batch sizes should reach the worker validation."""

    def _fake_worker(
        args: MorpionBootstrapArgs,
        *,
        max_nodes_per_patch: int = 10_000,
    ) -> MorpionReevaluationWorkerResult:
        del args
        if max_nodes_per_patch < 0:
            raise _negative_max_nodes_per_patch_error()
        return _make_reevaluation_worker_result()

    monkeypatch.setattr(
        launcher_module,
        "run_morpion_reevaluation_worker_once",
        _fake_worker,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "reevaluation",
            "--reevaluation-max-nodes-per-patch",
            "-1",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    with pytest.raises(ValueError, match="max_nodes_per_patch must be >= 0"):
        run_morpion_bootstrap_experiment(launcher_args)


def test_artifact_pipeline_dataset_worker_dispatches_autonomous_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset-worker CLI dispatch should call the autonomous worker only."""
    captured: list[MorpionBootstrapArgs] = []
    fake_result = _make_pipeline_worker_result("dataset")

    def _fake_worker(args: MorpionBootstrapArgs) -> MorpionPipelineWorkerResult:
        captured.append(args)
        return fake_result

    def _unexpected_runner(*args: object, **kwargs: object) -> object:
        raise _unexpected_pipeline_worker_runner_error()

    monkeypatch.setattr(
        launcher_module, "AnemoneMorpionSearchRunner", _unexpected_runner
    )
    monkeypatch.setattr(
        launcher_module,
        "run_next_pipeline_dataset_stage_once",
        _fake_worker,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "dataset_worker",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )
    result = run_morpion_bootstrap_experiment(launcher_args)

    assert result is fake_result
    assert len(captured) == 1
    assert captured[0].pipeline_mode == "artifact_pipeline"


def test_artifact_pipeline_worker_first_run_writes_bootstrap_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh worker launch should persist the canonical bootstrap config."""
    fake_result = _make_pipeline_worker_result("dataset")
    monkeypatch.setattr(
        launcher_module,
        "run_next_pipeline_dataset_stage_once",
        lambda args: fake_result,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "dataset_worker",
            "--min-visit-count",
            "7",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    assert run_morpion_bootstrap_experiment(launcher_args) is fake_result

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    assert paths.bootstrap_config_path.is_file()
    expected_args = replace(
        launcher_args.bootstrap_args,
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )
    assert load_bootstrap_config(
        paths.bootstrap_config_path
    ) == bootstrap_config_from_args(expected_args)


def test_dataset_worker_rejects_owned_persisted_config_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset workers should reject persisted dataset config drift."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    persisted_args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
        min_visit_count=1,
    )
    save_bootstrap_config(
        bootstrap_config_from_args(persisted_args),
        paths.bootstrap_config_path,
    )
    captured: list[MorpionBootstrapArgs] = []
    fake_result = _make_pipeline_worker_result("dataset")

    def _fake_worker(args: MorpionBootstrapArgs) -> MorpionPipelineWorkerResult:
        captured.append(args)
        return fake_result

    monkeypatch.setattr(
        launcher_module,
        "run_next_pipeline_dataset_stage_once",
        _fake_worker,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "dataset_worker",
            "--min-visit-count",
            "99",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    with pytest.raises(IncompatibleStageBootstrapConfigError, match="min_visit_count"):
        run_morpion_bootstrap_experiment(launcher_args)

    assert captured == []


def test_dataset_worker_rejects_foreign_persisted_config_difference(
    tmp_path: Path,
) -> None:
    """Dataset workers should ignore growth-only runtime config drift."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    persisted_args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
        tree_branch_limit=1000,
    )
    save_bootstrap_config(
        bootstrap_config_from_args(persisted_args),
        paths.bootstrap_config_path,
    )
    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "dataset_worker",
            "--tree-branch-limit",
            "1001",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    fake_result = _make_pipeline_worker_result("dataset")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            launcher_module,
            "run_next_pipeline_dataset_stage_once",
            lambda args: fake_result,
        )
        assert run_morpion_bootstrap_experiment(launcher_args) is fake_result
    finally:
        monkeypatch.undo()


def test_growth_stage_uses_requested_runtime_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growth relaunches should keep the requested per-invocation batch size."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    persisted_args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
        max_growth_steps_per_cycle=1000,
    )
    save_bootstrap_config(
        bootstrap_config_from_args(persisted_args),
        paths.bootstrap_config_path,
    )
    captured: list[MorpionBootstrapArgs] = []

    monkeypatch.setattr(launcher_module, "_build_launcher_runner", lambda _: object())

    def _fake_growth_stage(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_cycles: int,
    ) -> MorpionBootstrapRunState:
        del runner, max_cycles
        captured.append(args)
        return MorpionBootstrapRunState(
            generation=0,
            cycle_index=0,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=0,
            last_save_unix_s=None,
        )

    monkeypatch.setattr(
        launcher_module, "run_pipeline_growth_stage", _fake_growth_stage
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "growth",
            "--max-growth-steps-per-cycle",
            "10",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    run_morpion_bootstrap_experiment(launcher_args)

    assert len(captured) == 1
    assert captured[0].max_growth_steps_per_cycle == 10


def test_growth_stage_uses_requested_tree_branch_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growth relaunches should keep the requested tree branch limit."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    persisted_args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
        tree_branch_limit=1_000_000,
    )
    save_bootstrap_config(
        bootstrap_config_from_args(persisted_args),
        paths.bootstrap_config_path,
    )
    captured_runner_args: list[object] = []

    monkeypatch.setattr(
        launcher_module,
        "AnemoneMorpionSearchRunner",
        lambda runner_args: captured_runner_args.append(runner_args) or object(),
    )

    def _fake_growth_stage(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_cycles: int,
    ) -> MorpionBootstrapRunState:
        del runner, max_cycles
        return MorpionBootstrapRunState(
            generation=0,
            cycle_index=0,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=0,
            last_save_unix_s=None,
        )

    monkeypatch.setattr(
        launcher_module, "run_pipeline_growth_stage", _fake_growth_stage
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "growth",
            "--tree-branch-limit",
            "128",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )

    run_morpion_bootstrap_experiment(launcher_args)

    assert len(captured_runner_args) == 1
    assert (
        captured_runner_args[0].search_args.stopping_criterion.tree_branch_limit == 128
    )


def test_artifact_pipeline_training_worker_dispatches_autonomous_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training-worker CLI dispatch should call the autonomous worker only."""
    captured: list[MorpionBootstrapArgs] = []
    fake_result = _make_pipeline_worker_result("training")

    def _fake_worker(args: MorpionBootstrapArgs) -> MorpionPipelineWorkerResult:
        captured.append(args)
        return fake_result

    def _unexpected_runner(*args: object, **kwargs: object) -> object:
        raise _unexpected_pipeline_worker_runner_error()

    monkeypatch.setattr(
        launcher_module, "AnemoneMorpionSearchRunner", _unexpected_runner
    )
    monkeypatch.setattr(
        launcher_module,
        "run_next_pipeline_training_stage_once",
        _fake_worker,
    )

    launcher_args = launcher_module.launcher_args_from_cli(
        [
            "--work-dir",
            str(tmp_path),
            "--pipeline-mode",
            "artifact_pipeline",
            "--pipeline-stage",
            "training_worker",
            "--no-print-startup-summary",
            "--no-print-dashboard-hint",
        ]
    )
    result = run_morpion_bootstrap_experiment(launcher_args)

    assert result is fake_result
    assert len(captured) == 1
    assert captured[0].pipeline_mode == "artifact_pipeline"


@pytest.mark.parametrize(
    "pipeline_stage",
    [
        "growth",
        "dataset",
        "dataset_worker",
        "training",
        "training_worker",
        "reevaluation",
    ],
)
def test_single_process_rejects_non_loop_pipeline_stage(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    pipeline_stage: str,
) -> None:
    """Single-process CLI mode should reject non-loop pipeline stages."""
    with pytest.raises(SystemExit):
        launcher_module.launcher_args_from_cli(
            [
                "--work-dir",
                str(tmp_path),
                "--pipeline-stage",
                pipeline_stage,
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
