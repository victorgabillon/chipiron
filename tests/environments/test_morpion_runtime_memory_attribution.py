"""Diagnostic bounds, immutable restore, and current worker-memory behavior."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
from types import SimpleNamespace

import pytest

from chipiron.environments.morpion.bootstrap.config import bootstrap_config_from_dict
from chipiron.environments.morpion.bootstrap.derived.prepare import prepare_workspace
from chipiron.environments.morpion.bootstrap.derived.provenance import read_json, sha256
from chipiron.environments.morpion.bootstrap.derived.runtime import (
    args_from_config,
    make_runner,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_growth_stage,
)
from chipiron.environments.morpion.bootstrap.profiling.runtime_attribution.attribution import (
    ProfileLimits,
    account_components,
)
from chipiron.environments.morpion.bootstrap.profiling.runtime_attribution.checkpoint_audit import (
    inspect_checkpoint,
)
from chipiron.environments.morpion.bootstrap.profiling.runtime_attribution.cli import (
    RestoreOnlyRunner,
    main,
)
from tests.environments.test_morpion_derived_continuation import (
    _plan,
    historical,  # noqa: F401 - shared pytest fixture
)


@pytest.fixture
def prepared(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Reuse the existing tiny real checkpoint fixture, never historical user data."""
    return prepare_workspace(
        _plan(request.getfixturevalue("historical"), tmp_path / "derived")
    )


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}


def test_restore_only_is_immutable_and_reports_phases(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real tiny restore observes lazy/sparse state but cannot grow or save."""

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Diagnostic entered a scientific stage or checkpoint save")

    from anemone.tree_exploration import TreeExploration

    from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module

    original_logger_factory = runner_module.restore_memory_logger_for_checkpoint_path
    monkeypatch.setattr(
        runner_module, "load_morpion_search_checkpoint_payload", forbidden
    )
    monkeypatch.setattr(runner_module, "load_search_from_checkpoint_payload", forbidden)

    monkeypatch.setattr(TreeExploration, "step", forbidden)
    monkeypatch.setattr(RestoreOnlyRunner, "save_checkpoint", forbidden)
    before = _hashes(prepared)
    output = tmp_path / "profile"
    assert (
        main([
            "--work-dir",
            str(prepared),
            "--output-dir",
            str(output),
            "--restore-only",
            "--attach-evaluator",
            "--sample-nodes",
            "8",
            "--deep",
            "--recursive-max-objects",
            "100",
        ])
        == 0
    )
    assert _hashes(prepared) == before
    summary = read_json(output / "summary.json")
    assert summary["growth_steps"] == 0
    assert summary["attribution_added_decodes"] == 0
    assert summary["state_decodes"] == {"anchor": 1, "delta": 4}
    assert summary["scientific_artifacts_unchanged"]
    phases = [
        json.loads(line)
        for line in (output / "restore-phases.jsonl").read_text().splitlines()
    ]
    names = {item["phase"] for item in phases}
    assert {
        "after_manifest_load",
        "after_state_payloads_shard_load",
        "after_payload_store_build",
        "after_node_shells_shard_load",
        "after_state_handle_creation",
        "after_live_node_creation",
        "after_tree_bookkeeping",
        "after_edges_and_runtime_state_restored",
        "after_explicit_selector_restoration",
        "after_latest_expansions_restoration",
        "after_drop_raw_checkpoint_payload_if_applicable",
        "after_gc",
        "final_live_runtime",
    } <= names
    assert all(item["rss_mib"] > 0 and item["elapsed_s"] >= 0 for item in phases)
    profile = read_json(output / "cheap-profile.json")
    assert profile["sample_node_count"] <= 8
    assert profile["model_tensor_storage"]["cpu_bytes"] > 0
    assert profile["optimizations"]["payload_store_types"] == [
        "DenseCheckpointPayloadStore"
    ]
    assert set(profile["optimizations"]["sample_handle_types"]) == {
        "CheckpointBackedStateHandle"
    }
    assert profile["optimizations"]["rematerialization_cache_count"] == 0
    assert profile["optimizations"]["linoo_sample_default_entries"] == 0
    assert profile["optimizations"]["resolved_state_cache_counts"] == [5]
    assert profile["optimizations"]["linoo_sparse_table_counts"]
    assert profile["optimizations"]["lazy_evaluation_sample"]["node_evaluation_types"]
    deep = read_json(output / "bounded-deep-profile.json")
    assert deep["recursive"]["visited_objects"] <= 100
    assert (
        runner_module.restore_memory_logger_for_checkpoint_path
        is original_logger_factory
    )


def test_shallow_accounting_counts_shared_objects_once() -> None:
    """Obvious aliases must not inflate the sum of component measurements."""
    shared = [object(), object()]
    result = account_components(
        {"first": (shared,), "second": (shared,)}, limits=ProfileLimits()
    )
    assert result["components"]["first"]["shallow_object_count"] == 1
    assert result["components"]["second"]["shallow_bytes"] == 0
    assert (
        result["accounted_python_bytes"]
        == result["components"]["first"]["shallow_bytes"]
    )


def test_recursive_measurement_global_budget_and_no_materialization() -> None:
    """A shared cyclic structure consumes one budget without invoking state properties."""

    class Lazy:
        @property
        def state(self) -> object:
            pytest.fail("Profiler materialized a state")

    root = Lazy()
    root.children = [SimpleNamespace(values=list(range(50))) for _ in range(100)]
    result = account_components(
        {"one": (root,), "two": (root,)},
        limits=ProfileLimits(max_objects=12),
        deep=True,
    )
    assert 0 < result["recursive"]["visited_objects"] <= 12
    assert result["recursive"]["capped"]
    assert result["components"]["two"]["extra_reachable_bytes"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_nodes": 0},
        {"sample_nodes": 1000000},
        {"max_objects": 0},
        {"max_objects": 100001},
        {"max_depth": 99},
    ],
)
def test_unlimited_profile_options_refused(kwargs: dict) -> None:
    """Diagnostic memory caps cannot be disabled."""
    with pytest.raises(ValueError):
        ProfileLimits(**kwargs)


def test_restore_mode_requires_explicit_opt_in_and_external_output(
    prepared: Path,
) -> None:
    """No implicit execution and no reports inside the scientific workspace."""
    with pytest.raises(SystemExit):
        main(["--work-dir", str(prepared), "--output-dir", str(prepared / "profile")])
    before = _hashes(prepared)
    assert (
        main([
            "--work-dir",
            str(prepared),
            "--output-dir",
            str(prepared / "profile"),
            "--restore-only",
        ])
        == 2
    )
    assert _hashes(prepared) == before


def test_inspect_only_never_constructs_runner(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk evidence is separate from live runtime measurements."""

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Disk audit attempted runtime construction")

    monkeypatch.setattr(RestoreOnlyRunner, "__init__", forbidden)
    before = _hashes(prepared)
    out = tmp_path / "inspect"
    assert (
        main(["--work-dir", str(prepared), "--output-dir", str(out), "--inspect-only"])
        == 0
    )
    audit = read_json(out / "checkpoint-audit.json")
    assert audit["split_layout"] and audit["dense_zero_based_ids"]
    assert not audit["live_runtime_constructed"]
    assert _hashes(prepared) == before


def test_growth_and_checkpoint_save_are_forbidden() -> None:
    """The diagnostic runner rejects accidental execution without needing a runtime."""
    runner = object.__new__(RestoreOnlyRunner)
    with pytest.raises(RuntimeError, match="fresh creation"):
        runner.load_or_create(None, None)
    with pytest.raises(RuntimeError, match="Growth is forbidden"):
        runner.grow(1)
    with pytest.raises(RuntimeError, match="saving is forbidden"):
        runner.save_checkpoint("unused")


def test_split_checkpoint_audit_refuses_flat_fallback(prepared: Path) -> None:
    """A non-split manifest cannot silently select the typed-payload restore path."""
    path = prepared / "search_checkpoints/generation_000038.sharded"
    manifest = read_json(path / "manifest.json")
    manifest["shards"] = [s for s in manifest["shards"] if s["kind"] != "node_shells"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="split sharded"):
        inspect_checkpoint(path)


def test_multi_cycle_growth_rebuilds_runtime_every_cycle(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Document current behavior: retaining the runner does not retain its runtime."""
    config = bootstrap_config_from_dict(read_json(prepared / "bootstrap_config.json"))
    runner = make_runner(prepared, config, 123)
    args = args_from_config(prepared, config)
    originals: list[object] = []
    paths: list[Path] = []
    load = runner._load_runtime_from_checkpoint

    def observe(path: Path, **kwargs: object) -> object:
        runtime = load(path, **kwargs)
        originals.append(runtime)
        paths.append(path)
        return runtime

    monkeypatch.setattr(runner, "_load_runtime_from_checkpoint", observe)
    result = run_pipeline_growth_stage(args, runner, max_cycles=2)
    assert len(paths) == 2
    assert originals[0] is not originals[1]
    assert result.cycle_index == 44


def test_both_artifact_workers_load_complete_training_snapshots(prepared: Path) -> None:
    """Streaming row/patch output currently follows full snapshot materialization."""
    from chipiron.environments.morpion.bootstrap.cycle_dataset import (
        load_training_snapshot_for_generation,
    )
    from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
        load_reevaluation_training_tree_snapshot,
    )

    config = bootstrap_config_from_dict(read_json(prepared / "bootstrap_config.json"))
    args = args_from_config(prepared, config)
    path = prepared / "tree_exports_sharded/generation_000038.json"
    before = _hashes(prepared)
    expected = read_json(path)["node_count"]
    dataset = load_training_snapshot_for_generation(args=args, artifact_path=path)
    reevaluation = load_reevaluation_training_tree_snapshot(path)
    assert len(dataset.nodes) == len(reevaluation.nodes) == expected
    assert dataset is not reevaluation
    assert dataset.nodes is not reevaluation.nodes
    assert _hashes(prepared) == before


@pytest.mark.parametrize("available", [None, 0.0])
def test_ram_guard_refuses_restore(
    prepared: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: float | None,
) -> None:
    """Unknown or insufficient RAM cannot enter restore, even with a lower CLI floor."""
    from chipiron.environments.morpion.bootstrap.profiling.runtime_attribution import (
        cli,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("RAM guard allowed runtime construction")

    monkeypatch.setattr(cli, "available_ram_mb", lambda: available)
    monkeypatch.setattr(RestoreOnlyRunner, "__init__", forbidden)
    assert (
        main([
            "--work-dir",
            str(prepared),
            "--output-dir",
            str(tmp_path / "refused"),
            "--restore-only",
            "--minimum-free-before-restore-mib",
            "1",
        ])
        == 2
    )


def test_instrumentation_is_removed_after_exception(tmp_path: Path) -> None:
    """Diagnostic hooks cannot survive an interrupted restore."""
    from anemone.checkpoints import state_handles

    from chipiron.environments.morpion.bootstrap.profiling.runtime_attribution.phases import (
        PhaseRecorder,
        observe_restore,
    )

    original = state_handles.CheckpointStateResolver._resolve_anchor
    with (
        pytest.raises(RuntimeError, match="interrupt"),
        observe_restore(PhaseRecorder(tmp_path / "phases.jsonl")),
    ):
        assert state_handles.CheckpointStateResolver._resolve_anchor is not original
        raise RuntimeError("interrupt")
    assert state_handles.CheckpointStateResolver._resolve_anchor is original
