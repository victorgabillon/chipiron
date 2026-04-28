"""Tests for the Morpion reevaluation patch producer."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

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

import chipiron.environments.morpion.bootstrap.reevaluation_worker as reevaluation_worker_module
from chipiron.environments.morpion.bootstrap import (
    MorpionActiveModelNodeReevaluationEvaluator,
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionNodeReevaluationEvaluator,
    MorpionPipelineActiveModel,
    MorpionPipelineGenerationManifest,
    MorpionReevaluationCursor,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    MorpionReevaluationWorkerResult,
    build_active_model_reevaluation_evaluator,
    cursor_matches_active_model,
    load_reevaluation_cursor,
    load_reevaluation_patch,
    resolve_latest_reevaluation_tree_snapshot,
    run_morpion_reevaluation_worker_once,
    save_pipeline_active_model,
    save_pipeline_manifest,
    save_reevaluation_cursor,
    save_reevaluation_patch,
    select_reevaluation_node_window,
    snapshot_values_to_patch_rows,
)


class FakeNodeReevaluationEvaluator:
    """Deterministic test double for injected reevaluation worker paths."""

    def evaluate_patch_rows(
        self,
        snapshot: TrainingTreeSnapshot,
        node_ids: Sequence[str],
    ) -> tuple[MorpionReevaluationPatchRow, ...]:
        """Return deterministic patch rows for the selected node ids."""
        del snapshot
        return tuple(
            MorpionReevaluationPatchRow(
                node_id=node_id,
                direct_value=100.0 + float(index),
                backed_up_value=None,
                metadata={"source": "fake_injected_evaluator"},
            )
            for index, node_id in enumerate(node_ids)
        )


@dataclass(frozen=True, slots=True)
class _StubValue:
    score: float


class _StubLoadedEvaluator:
    """Tiny state evaluator used to test the real adapter without model inference."""

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = tuple(scores)
        self.calls: list[object] = []

    def evaluate(self, state: object) -> _StubValue:
        self.calls.append(state)
        return _StubValue(score=self._scores[len(self.calls) - 1])


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return codec.dump_state_ref(next_state)


def _make_training_snapshot(node_ids: tuple[str, ...]) -> TrainingTreeSnapshot:
    """Build one training snapshot with deterministic node values."""
    payload = _make_morpion_payload()
    return TrainingTreeSnapshot(
        root_node_id=node_ids[0] if node_ids else "root",
        nodes=tuple(
            TrainingNodeSnapshot(
                node_id=node_id,
                parent_ids=(),
                child_ids=(),
                depth=index,
                state_ref_payload=payload,
                direct_value_scalar=float(index) + 0.25,
                backed_up_value_scalar=float(index) + 0.5,
                is_terminal=index % 2 == 0,
                is_exact=index % 2 == 1,
                over_event_label=None,
                visit_count=7 + index,
                metadata={"source": "reevaluation-worker-test"},
            )
            for index, node_id in enumerate(node_ids)
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


def _artifact_pipeline_args(work_dir: Path) -> MorpionBootstrapArgs:
    """Build one small artifact-pipeline arg set for reevaluation tests."""
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


def _write_active_model(
    paths: MorpionBootstrapPaths,
    *,
    generation: int = 2,
    evaluator_name: str = "default",
    model_bundle_path: str = "models/generation_000002/default",
) -> MorpionPipelineActiveModel:
    """Persist one pipeline active-model record for worker tests."""
    active_model = MorpionPipelineActiveModel(
        generation=generation,
        evaluator_name=evaluator_name,
        model_bundle_path=model_bundle_path,
        updated_at_utc="2026-04-28T12:00:00Z",
        metadata={"source": "reevaluation-worker-test"},
    )
    save_pipeline_active_model(active_model, paths.pipeline_active_model_path)
    return active_model


def _write_manifest_with_snapshot(
    paths: MorpionBootstrapPaths,
    *,
    generation: int,
    snapshot: TrainingTreeSnapshot,
) -> Path:
    """Persist one manifest and its training snapshot for worker tests."""
    snapshot_path = paths.tree_snapshot_path_for_generation(generation)
    save_training_tree_snapshot(snapshot, snapshot_path)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=generation,
            created_at_utc="2026-04-28T12:00:00Z",
            tree_snapshot_path=paths.relative_to_work_dir(snapshot_path),
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(generation),
    )
    return snapshot_path


def test_select_reevaluation_node_window_contiguous_without_wrap() -> None:
    """One bounded window should advance contiguously when no wrap is needed."""
    selected, next_cursor, completed_full_pass = select_reevaluation_node_window(
        ("a", "b", "c", "d"),
        start_cursor="b",
        max_nodes=2,
    )

    assert selected == ("b", "c")
    assert next_cursor == "d"
    assert not completed_full_pass


def test_select_reevaluation_node_window_wraps_once() -> None:
    """One bounded window should wrap exactly once when the slice crosses the end."""
    selected, next_cursor, completed_full_pass = select_reevaluation_node_window(
        ("a", "b", "c", "d"),
        start_cursor="c",
        max_nodes=3,
    )

    assert selected == ("c", "d", "a")
    assert next_cursor == "b"
    assert completed_full_pass


def test_select_reevaluation_node_window_caps_at_all_nodes() -> None:
    """One large request should include each node at most once and wrap to the first id."""
    selected, next_cursor, completed_full_pass = select_reevaluation_node_window(
        ("a", "b", "c", "d"),
        start_cursor="a",
        max_nodes=10,
    )

    assert selected == ("a", "b", "c", "d")
    assert next_cursor == "a"
    assert completed_full_pass
    assert len(set(selected)) == len(selected)


def test_worker_returns_missing_active_model(tmp_path: Path) -> None:
    """The worker should skip cleanly when no active evaluator exists yet."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    result = run_morpion_reevaluation_worker_once(_artifact_pipeline_args(tmp_path))

    assert result == MorpionReevaluationWorkerResult(
        patch_written=False,
        reason="missing_active_model",
        patch_id=None,
        num_rows=0,
        evaluator_generation=None,
        evaluator_name=None,
        start_cursor=None,
        end_cursor=None,
        completed_full_pass_count=None,
    )
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()


def test_worker_returns_missing_tree_snapshot(tmp_path: Path) -> None:
    """The worker should skip cleanly when no usable tree snapshot exists yet."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    _write_active_model(paths)

    result = run_morpion_reevaluation_worker_once(_artifact_pipeline_args(tmp_path))

    assert not result.patch_written
    assert result.reason == "missing_tree_snapshot"
    assert result.evaluator_generation == 2
    assert result.evaluator_name == "default"
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()


def test_worker_writes_patch_and_cursor(tmp_path: Path) -> None:
    """The worker should write one bounded patch and advance the reevaluation cursor."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("node-c", "node-a", "node-b")),
    )

    result = run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(tmp_path),
        max_nodes_per_patch=2,
        now_unix_s=1_777_377_600.0,
        patch_id="patch-1",
        use_snapshot_value_fallback=True,
    )

    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    cursor = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)

    assert result.patch_written
    assert result.reason is None
    assert result.patch_id == "patch-1"
    assert result.num_rows == 2
    assert result.evaluator_generation == active_model.generation
    assert result.evaluator_name == active_model.evaluator_name
    assert result.start_cursor == "node-a"
    assert result.end_cursor == "node-b"
    assert result.completed_full_pass_count == 0
    assert patch.patch_id == "patch-1"
    assert patch.evaluator_generation == active_model.generation
    assert patch.evaluator_name == active_model.evaluator_name
    assert patch.model_bundle_path == active_model.model_bundle_path
    assert patch.tree_generation == 7
    assert tuple(row.node_id for row in patch.rows) == ("node-a", "node-b")
    assert patch.metadata["max_nodes_per_patch"] == 2
    assert patch.metadata["next_node_cursor"] == "node-c"
    assert cursor.next_node_cursor == "node-c"
    assert cursor.completed_full_pass_count == 0
    assert cursor.last_patch_id == "patch-1"


def test_build_active_model_reevaluation_evaluator_resolves_relative_path(
    tmp_path: Path,
) -> None:
    """The real reevaluation adapter builder should resolve model paths from work dir."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    expected_bundle_path = cast(
        "Path",
        paths.resolve_work_dir_path(active_model.model_bundle_path),
    )
    expected_bundle_path.mkdir(parents=True)

    evaluator = build_active_model_reevaluation_evaluator(
        paths=paths,
        active_model=active_model,
    )

    assert isinstance(evaluator, MorpionActiveModelNodeReevaluationEvaluator)
    assert evaluator.model_bundle_path == expected_bundle_path


def test_build_active_model_reevaluation_evaluator_rejects_missing_bundle(
    tmp_path: Path,
) -> None:
    """The real reevaluation adapter builder should fail loudly for missing bundles."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(
        paths,
        model_bundle_path="models/generation_000002/missing-default",
    )

    with pytest.raises(
        FileNotFoundError,
        match="models/generation_000002/missing-default",
    ):
        build_active_model_reevaluation_evaluator(
            paths=paths,
            active_model=active_model,
        )


def test_active_model_reevaluation_evaluator_uses_model_and_terminal_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real adapter should preserve exact terminal values and model-evaluate others."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    bundle_path = cast("Path", paths.resolve_work_dir_path(active_model.model_bundle_path))
    bundle_path.mkdir(parents=True)
    snapshot = _make_training_snapshot(("a", "b"))
    stub_evaluator = _StubLoadedEvaluator((42.5,))

    monkeypatch.setattr(
        reevaluation_worker_module,
        "load_morpion_evaluator_from_model_bundle",
        lambda model_bundle_path: stub_evaluator,
    )

    evaluator = build_active_model_reevaluation_evaluator(
        paths=paths,
        active_model=active_model,
    )
    rows = evaluator.evaluate_patch_rows(snapshot, ("a", "b"))

    assert rows[0].direct_value == pytest.approx(0.5)
    assert rows[0].backed_up_value is None
    assert rows[0].metadata["source"] == "terminal_existing_value"
    assert rows[0].metadata["model_bundle_path"] == str(bundle_path)
    assert rows[1].direct_value == pytest.approx(42.5)
    assert rows[1].metadata["source"] == "active_model_reevaluation"
    assert rows[1].metadata["model_bundle_path"] == str(bundle_path)
    assert len(stub_evaluator.calls) == 1


def test_worker_uses_injected_evaluator(tmp_path: Path) -> None:
    """Injected reevaluation evaluators should override the default active-model path."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("node-c", "node-a", "node-b")),
    )

    result = run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(tmp_path),
        evaluator=FakeNodeReevaluationEvaluator(),
        max_nodes_per_patch=2,
        now_unix_s=1_777_377_600.0,
        patch_id="patch-fake",
    )

    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)

    assert result.patch_written
    assert result.evaluator_generation == active_model.generation
    assert tuple(row.node_id for row in patch.rows) == ("node-a", "node-b")
    assert tuple(row.direct_value for row in patch.rows) == (100.0, 101.0)
    assert patch.rows[0].metadata["source"] == "fake_injected_evaluator"


def test_worker_default_path_fails_loudly_for_missing_bundle(tmp_path: Path) -> None:
    """The default worker path should use the active-model adapter, not silent fallback."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    _write_active_model(paths, model_bundle_path="models/generation_000002/missing")
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("a", "b", "c")),
    )

    with pytest.raises(FileNotFoundError, match="models/generation_000002/missing"):
        run_morpion_reevaluation_worker_once(_artifact_pipeline_args(tmp_path))


def test_worker_does_not_overwrite_pending_patch(tmp_path: Path) -> None:
    """The worker should skip cleanly when a pending patch already exists."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("a", "b", "c")),
    )
    existing_patch = MorpionReevaluationPatch(
        patch_id="existing-patch",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=active_model.generation,
        evaluator_name=active_model.evaluator_name,
        model_bundle_path=active_model.model_bundle_path,
        rows=(
            MorpionReevaluationPatchRow(
                node_id="a",
                direct_value=0.25,
                metadata={"source": "existing"},
            ),
        ),
        tree_generation=7,
        start_cursor="a",
        end_cursor="a",
        metadata={"source": "existing"},
    )
    save_reevaluation_patch(existing_patch, paths.pipeline_reevaluation_patch_path)
    existing_cursor = MorpionReevaluationCursor(
        evaluator_generation=active_model.generation,
        evaluator_name=active_model.evaluator_name,
        model_bundle_path=active_model.model_bundle_path,
        next_node_cursor="b",
        updated_at_utc="2026-04-28T12:00:00Z",
        tree_generation=7,
        completed_full_pass_count=0,
        last_patch_id="existing-patch",
        metadata={"source": "existing"},
    )
    save_reevaluation_cursor(existing_cursor, paths.pipeline_reevaluation_cursor_path)

    result = run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(tmp_path),
        max_nodes_per_patch=2,
        patch_id="new-patch",
    )

    assert not result.patch_written
    assert result.reason == "pending_patch_exists"
    assert load_reevaluation_patch(paths.pipeline_reevaluation_patch_path) == existing_patch
    assert load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path) == existing_cursor


def test_worker_does_not_advance_cursor_when_patch_race_is_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late competing patch should win without advancing this worker's cursor."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("a", "b", "c")),
    )
    existing_cursor = MorpionReevaluationCursor(
        evaluator_generation=active_model.generation,
        evaluator_name=active_model.evaluator_name,
        model_bundle_path=active_model.model_bundle_path,
        next_node_cursor="b",
        updated_at_utc="2026-04-28T12:00:00Z",
        tree_generation=7,
        completed_full_pass_count=2,
        last_patch_id="old-patch",
        metadata={"source": "existing"},
    )
    save_reevaluation_cursor(existing_cursor, paths.pipeline_reevaluation_cursor_path)
    competing_patch = MorpionReevaluationPatch(
        patch_id="competing-patch",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=active_model.generation,
        evaluator_name=active_model.evaluator_name,
        model_bundle_path=active_model.model_bundle_path,
        rows=(
            MorpionReevaluationPatchRow(
                node_id="b",
                direct_value=1.25,
                metadata={"source": "competing"},
            ),
        ),
        tree_generation=7,
        start_cursor="b",
        end_cursor="b",
        metadata={"source": "competing"},
    )

    def lose_race(
        patch: MorpionReevaluationPatch,
        path: Path,
    ) -> bool:
        assert patch.patch_id == "new-patch"
        save_reevaluation_patch(competing_patch, path)
        return False

    monkeypatch.setattr(
        reevaluation_worker_module,
        "_save_reevaluation_patch_exclusive",
        lose_race,
    )

    result = run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(tmp_path),
        max_nodes_per_patch=2,
        patch_id="new-patch",
        use_snapshot_value_fallback=True,
    )

    assert not result.patch_written
    assert result.reason == "pending_patch_exists"
    assert load_reevaluation_patch(paths.pipeline_reevaluation_patch_path) == competing_patch
    assert load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path) == existing_cursor


def test_worker_continues_from_cursor_and_wraps(tmp_path: Path) -> None:
    """The worker should continue from the persisted cursor and wrap cleanly."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("a", "b", "c", "d")),
    )
    save_reevaluation_cursor(
        MorpionReevaluationCursor(
            evaluator_generation=active_model.generation,
            evaluator_name=active_model.evaluator_name,
            model_bundle_path=active_model.model_bundle_path,
            next_node_cursor="c",
            updated_at_utc="2026-04-28T12:00:00Z",
            tree_generation=7,
            completed_full_pass_count=0,
            last_patch_id=None,
            metadata={"source": "existing"},
        ),
        paths.pipeline_reevaluation_cursor_path,
    )

    result = run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(tmp_path),
        max_nodes_per_patch=3,
        now_unix_s=1_777_377_600.0,
        patch_id="patch-2",
        use_snapshot_value_fallback=True,
    )

    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    cursor = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)

    assert result.patch_written
    assert tuple(row.node_id for row in patch.rows) == ("c", "d", "a")
    assert patch.start_cursor == "c"
    assert patch.end_cursor == "a"
    assert cursor.next_node_cursor == "b"
    assert cursor.completed_full_pass_count == 1
    assert result.completed_full_pass_count == 1


def test_worker_resets_cursor_when_active_model_changes(tmp_path: Path) -> None:
    """The worker should reset the cursor when the active evaluator changes."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    _write_active_model(paths, generation=2)
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("b", "a", "c")),
    )
    save_reevaluation_cursor(
        MorpionReevaluationCursor(
            evaluator_generation=1,
            evaluator_name="default",
            model_bundle_path="models/generation_000001/default",
            next_node_cursor="c",
            updated_at_utc="2026-04-28T12:00:00Z",
            tree_generation=6,
            completed_full_pass_count=4,
            last_patch_id="old-patch",
            metadata={"source": "existing"},
        ),
        paths.pipeline_reevaluation_cursor_path,
    )

    result = run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(tmp_path),
        max_nodes_per_patch=1,
        now_unix_s=1_777_377_600.0,
        patch_id="patch-3",
        use_snapshot_value_fallback=True,
    )

    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    cursor = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)

    assert result.patch_written
    assert patch.start_cursor == "a"
    assert tuple(row.node_id for row in patch.rows) == ("a",)
    assert cursor.next_node_cursor == "b"
    assert cursor.completed_full_pass_count == 0
    assert result.completed_full_pass_count == 0


def test_worker_rejects_negative_max_nodes(tmp_path: Path) -> None:
    """Negative max-nodes settings should fail clearly."""
    with pytest.raises(ValueError, match="max_nodes_per_patch must be >= 0"):
        run_morpion_reevaluation_worker_once(
            _artifact_pipeline_args(tmp_path),
            max_nodes_per_patch=-1,
        )


def test_worker_zero_max_nodes_returns_without_writing(tmp_path: Path) -> None:
    """Zero max-nodes settings should skip cleanly without writing artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    active_model = _write_active_model(paths)
    _write_manifest_with_snapshot(
        paths,
        generation=7,
        snapshot=_make_training_snapshot(("a", "b", "c")),
    )
    existing_cursor = MorpionReevaluationCursor(
        evaluator_generation=active_model.generation,
        evaluator_name=active_model.evaluator_name,
        model_bundle_path=active_model.model_bundle_path,
        next_node_cursor="b",
        updated_at_utc="2026-04-28T12:00:00Z",
        tree_generation=7,
        completed_full_pass_count=2,
        last_patch_id="old-patch",
        metadata={"source": "existing"},
    )
    save_reevaluation_cursor(existing_cursor, paths.pipeline_reevaluation_cursor_path)

    result = run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(tmp_path),
        max_nodes_per_patch=0,
    )

    assert not result.patch_written
    assert result.reason == "max_nodes_per_patch_zero"
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path) == existing_cursor


def test_package_root_reexports_reevaluation_worker_api() -> None:
    """Package root should re-export the reevaluation worker public API."""
    assert (
        MorpionReevaluationWorkerResult
        is reevaluation_worker_module.MorpionReevaluationWorkerResult
    )
    assert (
        MorpionActiveModelNodeReevaluationEvaluator
        is reevaluation_worker_module.MorpionActiveModelNodeReevaluationEvaluator
    )
    assert (
        MorpionNodeReevaluationEvaluator
        is reevaluation_worker_module.MorpionNodeReevaluationEvaluator
    )
    assert (
        build_active_model_reevaluation_evaluator
        is reevaluation_worker_module.build_active_model_reevaluation_evaluator
    )
    assert (
        run_morpion_reevaluation_worker_once
        is reevaluation_worker_module.run_morpion_reevaluation_worker_once
    )
    assert (
        resolve_latest_reevaluation_tree_snapshot
        is reevaluation_worker_module.resolve_latest_reevaluation_tree_snapshot
    )
    assert (
        select_reevaluation_node_window
        is reevaluation_worker_module.select_reevaluation_node_window
    )
    assert (
        snapshot_values_to_patch_rows
        is reevaluation_worker_module.snapshot_values_to_patch_rows
    )
    assert (
        cursor_matches_active_model
        is reevaluation_worker_module.cursor_matches_active_model
    )
