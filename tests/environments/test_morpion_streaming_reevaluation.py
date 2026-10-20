"""Legacy/indexed worker differential tests, using isolated synthetic exports."""
# ruff: noqa: TRY003

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import chipiron.environments.morpion.bootstrap.reevaluation_publication as publication
import chipiron.environments.morpion.bootstrap.reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    MorpionReevaluationCursor,
    MorpionReevaluationPatchRow,
    load_reevaluation_cursor,
    save_reevaluation_cursor,
)
from chipiron.environments.morpion.bootstrap.launcher import launcher_args_from_cli
from tests.environments.test_morpion_bootstrap_reevaluation_worker import (
    _artifact_pipeline_args,
    _make_morpion_delta_payload,
    _make_training_snapshot,
    _StubLoadedEvaluator,
    _write_active_model,
    _write_manifest_with_sharded_snapshot,
    _write_manifest_with_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot


def setup_worker(
    root: Path, *, flat: bool = False, empty: bool = False
) -> MorpionBootstrapPaths:
    """Set up a published generation and fixed model identity without loading models."""
    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    _write_active_model(paths)
    snapshot = _make_training_snapshot(() if empty else ("2", "1", "20", "10", "100"))
    writer = (
        _write_manifest_with_snapshot if flat else _write_manifest_with_sharded_snapshot
    )
    writer(paths, generation=3, snapshot=snapshot)
    save_reevaluation_cursor(
        MorpionReevaluationCursor(
            evaluator_generation=2,
            evaluator_name="default",
            model_bundle_path="models/generation_000002/default",
            next_node_cursor="20",
            tree_generation=1,
            updated_at_utc="2026-04-28T12:00:00Z",
            completed_full_pass_count=4,
        ),
        paths.pipeline_reevaluation_cursor_path,
    )
    return paths


def run_once(
    paths: MorpionBootstrapPaths, *, streaming: bool = True, **kwargs: object
) -> worker.MorpionReevaluationWorkerResult:
    """Run a fixed synthetic pass; fixture timestamps and patch IDs are identical."""
    return worker.run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(paths.work_dir),
        use_streaming_sharded_reader=streaming,
        reader_cache_dir=paths.work_dir / "reader-cache",
        max_nodes_per_patch=3,
        now_unix_s=123456.0,
        patch_id="fixed-patch",
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("cursor", [None, "missing", "1", "20"])
def test_serialized_patch_cursor_and_result_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool, cursor: str | None
) -> None:
    """Compare all scientific metadata, rows, counts, ownership and worker status."""
    legacy = setup_worker(tmp_path / "legacy")
    bounded = setup_worker(tmp_path / "bounded")
    common_bundle = tmp_path / "bundle"
    common_bundle.mkdir()
    adapters: list[worker.MorpionActiveModelNodeReevaluationEvaluator] = []

    def build(**_: object) -> worker.MorpionActiveModelNodeReevaluationEvaluator:
        adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(common_bundle)
        adapter._evaluator = _StubLoadedEvaluator([100.25] * 10)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(worker, "build_active_model_reevaluation_evaluator", build)
    for paths in (legacy, bounded):
        persisted = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)
        save_reevaluation_cursor(
            replace(persisted, next_node_cursor=cursor),
            paths.pipeline_reevaluation_cursor_path,
        )
    expected = run_once(legacy, streaming=False, use_snapshot_value_fallback=fallback)
    actual = run_once(bounded, use_snapshot_value_fallback=fallback)
    assert actual == expected
    assert json.loads(
        bounded.pipeline_reevaluation_patch_path.read_text()
    ) == json.loads(legacy.pipeline_reevaluation_patch_path.read_text())
    assert json.loads(
        bounded.pipeline_reevaluation_cursor_path.read_text()
    ) == json.loads(legacy.pipeline_reevaluation_cursor_path.read_text())
    if not fallback:
        assert len(adapters) == 2
        assert len(adapters[0]._evaluator.calls) == len(adapters[1]._evaluator.calls)  # type: ignore[union-attr]
    observations = list(
        (bounded.work_dir / "pipeline" / "performance").rglob("reevaluation-*.json")
    )
    observation = json.loads(observations[0].read_text())
    assert observation["reader_mode"] == "sharded_indexed"
    assert observation["node_count"] == 5
    assert observation["rows_requested"] == 3
    assert "cursor_publication" in observation


def test_bounded_worker_never_loads_complete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forbid both full loaders and any TrainingTreeSnapshot construction."""
    paths = setup_worker(tmp_path)

    def forbidden(*_: object, **__: object) -> object:
        pytest.fail("full snapshot used by bounded worker")

    monkeypatch.setattr(worker, "load_reevaluation_training_tree_snapshot", forbidden)
    monkeypatch.setattr(worker, "TrainingTreeSnapshot", forbidden)
    assert run_once(paths, use_snapshot_value_fallback=True).patch_written


class CompleteSnapshotEvaluator:
    """A legacy evaluator asserts that it receives all nodes and original metadata."""

    def evaluate_patch_rows(
        self, snapshot: TrainingTreeSnapshot, node_ids: Sequence[str]
    ) -> tuple[MorpionReevaluationPatchRow, ...]:
        """Depend on complete-snapshot semantics for compatibility testing."""
        assert len(snapshot.nodes) == 5
        assert snapshot.root_node_id == "2"
        assert snapshot.metadata["source_generation"] == 3
        return worker.snapshot_values_to_patch_rows(snapshot, node_ids)


class WindowEvaluator:
    """New adapters explicitly opt in by implementing the separate window method."""

    def evaluate_node_window(
        self, nodes: Sequence[TrainingNodeSnapshot]
    ) -> tuple[MorpionReevaluationPatchRow, ...]:
        """Require exactly the selected ordered window."""
        assert tuple(node.node_id for node in nodes) == ("20", "1", "10")
        return worker.node_values_to_patch_rows(nodes)


def test_legacy_custom_evaluator_falls_back_observably(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Opting into the reader must not silently truncate legacy evaluator inputs."""
    paths = setup_worker(tmp_path)
    with caplog.at_level("INFO"):
        assert run_once(paths, evaluator=CompleteSnapshotEvaluator()).patch_written
    assert "custom_evaluator_requires_complete_snapshot" in caplog.text
    assert '"reader_mode": "legacy_full_snapshot"' in caplog.text


def test_window_evaluator_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Window protocol can be used without implementing the legacy protocol."""
    paths = setup_worker(tmp_path)
    monkeypatch.setattr(
        worker,
        "load_reevaluation_training_tree_snapshot",
        lambda *_: pytest.fail("legacy load"),
    )
    assert run_once(paths, evaluator=WindowEvaluator()).patch_written


def test_flat_snapshot_falls_back_observably(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Flat artifact semantics remain supported through the complete loader."""
    paths = setup_worker(tmp_path, flat=True)
    with caplog.at_level("INFO"):
        assert run_once(paths, use_snapshot_value_fallback=True).patch_written
    assert "legacy_flat_snapshot" in caplog.text
    assert '"reader_mode": "legacy_full_snapshot"' in caplog.text


@pytest.mark.parametrize(
    "changed_field",
    ["evaluator_generation", "evaluator_name", "model_bundle_path", "tree_generation"],
)
def test_cursor_ownership_policy_and_pass_count(
    tmp_path: Path, changed_field: str
) -> None:
    """Only model identity resets; advancing a tree preserves an existing cursor."""
    paths = setup_worker(tmp_path)
    old = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)
    value: object = 999 if changed_field.endswith("generation") else "different"
    save_reevaluation_cursor(
        replace(old, **{changed_field: value}), paths.pipeline_reevaluation_cursor_path
    )  # type: ignore[arg-type]
    result = run_once(paths, use_snapshot_value_fallback=True)
    if changed_field == "tree_generation":
        assert (
            result.start_cursor,
            result.end_cursor,
            result.completed_full_pass_count,
        ) == ("20", "10", 5)
    else:
        assert (
            result.start_cursor,
            result.end_cursor,
            result.completed_full_pass_count,
        ) == ("1", "100", 0)


@pytest.mark.parametrize(
    "race", ["before_inference", "after_inference", "exclusive_create", "write_failure"]
)
def test_pending_patch_and_failed_publication_never_advance_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    """Preserve both pending checks, exclusive-create race and failed-write guard."""
    paths = setup_worker(tmp_path)
    before = paths.pipeline_reevaluation_cursor_path.read_bytes()
    pending = paths.pipeline_reevaluation_patch_path

    class RacingEvaluator(WindowEvaluator):
        """Publish a competing pending patch at a controlled worker boundary."""

        def evaluate_node_window(
            self, nodes: Sequence[TrainingNodeSnapshot]
        ) -> tuple[MorpionReevaluationPatchRow, ...]:
            if race == "after_inference":
                pending.write_text("competitor")
            return super().evaluate_node_window(nodes)

    if race == "before_inference":
        pending.write_text("competitor")
        monkeypatch.setattr(
            worker,
            "ShardedTrainingGenerationView",
            lambda *_a, **_kw: pytest.fail("reader opened while blocked"),
        )
    elif race == "exclusive_create":
        save = worker._save_reevaluation_patch_exclusive

        def competing_save(patch: object, path: Path) -> bool:
            pending.write_text("competitor")
            return save(patch, path)  # type: ignore[arg-type]

        monkeypatch.setattr(
            worker, "_save_reevaluation_patch_exclusive", competing_save
        )
    elif race == "write_failure":

        def failed_fsync(_descriptor: int) -> None:
            raise OSError("synthetic write failure")

        monkeypatch.setattr(publication.os, "fsync", failed_fsync)
    if race == "write_failure":
        with pytest.raises(OSError, match="synthetic"):
            run_once(paths, evaluator=RacingEvaluator())
    else:
        result = run_once(paths, evaluator=RacingEvaluator())
        assert not result.patch_written
        assert result.reason == "pending_patch_exists"
        assert pending.read_text() == "competitor"
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == before


@pytest.mark.parametrize(
    "guard", ["ram", "zero", "missing_model", "missing_snapshot", "empty"]
)
def test_early_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: str
) -> None:
    """The streaming opt-in preserves worker idle guards and cursor contents."""
    paths = setup_worker(tmp_path, empty=guard == "empty")
    before = paths.pipeline_reevaluation_cursor_path.read_bytes()
    expected = {
        "ram": "low_available_ram",
        "zero": "max_nodes_per_patch_zero",
        "missing_model": "missing_active_model",
        "missing_snapshot": "missing_tree_snapshot",
        "empty": "empty_tree_snapshot",
    }[guard]
    if guard == "ram":
        monkeypatch.setattr(worker, "log_available_ram_guard", lambda **_: False)
    elif guard == "missing_model":
        paths.pipeline_active_model_path.unlink()
    elif guard == "missing_snapshot":
        paths.sharded_tree_snapshot_path_for_generation(3).unlink()
    if guard != "empty":
        monkeypatch.setattr(
            worker,
            "ShardedTrainingGenerationView",
            lambda *_a, **_kw: pytest.fail("reader before guard"),
        )
    result = worker.run_morpion_reevaluation_worker_once(
        _artifact_pipeline_args(paths.work_dir),
        max_nodes_per_patch=0 if guard == "zero" else 3,
        use_streaming_sharded_reader=True,
        use_snapshot_value_fallback=True,
        reader_cache_dir=tmp_path / "cache",
    )
    assert result.reason == expected
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == before


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_invalid_inference_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    """Finite-value validation still runs through the original score extractor."""
    paths = setup_worker(tmp_path)
    before = paths.pipeline_reevaluation_cursor_path.read_bytes()
    adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "bundle")
    adapter._evaluator = _StubLoadedEvaluator([bad_value])
    monkeypatch.setattr(
        worker, "build_active_model_reevaluation_evaluator", lambda **_: adapter
    )
    with pytest.raises(ValueError, match="non-finite"):
        run_once(paths)
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == before


def test_terminal_values_and_delta_codec_rejection(tmp_path: Path) -> None:
    """Terminal backups bypass inference; delta refs are never silently decoded."""
    nodes = _make_training_snapshot(("1", "2")).nodes
    adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "bundle")
    loaded = _StubLoadedEvaluator([12.5])
    adapter._evaluator = loaded
    terminal = replace(
        nodes[0],
        is_terminal=True,
        backed_up_value_scalar=9.75,
        state_ref_payload=_make_morpion_delta_payload(),
    )
    rows = adapter.evaluate_node_window((terminal, nodes[1]))
    assert rows[0].direct_value == 9.75
    assert rows[0].backed_up_value is None
    assert rows[0].metadata["source"] == "terminal_existing_value"
    assert rows[1].direct_value == 12.5
    assert len(loaded.calls) == 1
    with pytest.raises((TypeError, ValueError)):
        adapter.evaluate_node_window((replace(terminal, is_terminal=False),))


def test_launcher_opt_in_is_ephemeral_and_default_off(tmp_path: Path) -> None:
    """The new switch lives only in launcher args, outside scientific config."""
    default = launcher_args_from_cli(["--work-dir", str(tmp_path)])
    enabled = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--reevaluation-streaming-sharded-reader",
    ])
    assert not default.reevaluation_streaming_sharded_reader
    assert enabled.reevaluation_streaming_sharded_reader
    assert default.bootstrap_args == enabled.bootstrap_args


def test_launcher_forwards_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag reaches the worker, while no search runner or real worker launches."""
    import chipiron.environments.morpion.bootstrap.launcher as launcher

    captured: list[bool] = []
    expected = worker.MorpionReevaluationWorkerResult(
        patch_written=False,
        reason="synthetic",
        patch_id=None,
        num_rows=0,
        evaluator_generation=None,
        evaluator_name=None,
        start_cursor=None,
        end_cursor=None,
        completed_full_pass_count=None,
    )

    def fake_worker(
        *_: object, use_streaming_sharded_reader: bool = False, **__: object
    ) -> worker.MorpionReevaluationWorkerResult:
        captured.append(use_streaming_sharded_reader)
        return expected

    monkeypatch.setattr(launcher, "run_morpion_reevaluation_worker_once", fake_worker)
    args = launcher.launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--pipeline-mode",
        "artifact_pipeline",
        "--pipeline-stage",
        "reevaluation",
        "--reevaluation-streaming-sharded-reader",
        "--no-print-startup-summary",
        "--no-print-dashboard-hint",
    ])
    assert launcher.run_morpion_bootstrap_experiment(args) is expected
    assert captured == [True]
