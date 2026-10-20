"""Committed-cycle reuse and cold-reference equivalence on real small searches."""

from __future__ import annotations

import gc
import json
import weakref
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
from unittest.mock import Mock

import pytest
import zstandard

from chipiron.environments.morpion.bootstrap import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionBootstrapRolloutConfig,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineActiveModel,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    save_pipeline_active_model,
    save_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime.runner import default_search_args
from chipiron.environments.morpion.bootstrap.runtime.warm_growth import (
    GrowthRuntimeIdentityError,
    artifact_identity,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
    save_morpion_model_bundle,
)


def args_for(path: Path, *, save: bool = True) -> MorpionBootstrapArgs:
    """Keep all scientific fixtures small and isolated."""
    return MorpionBootstrapArgs(
        work_dir=path,
        pipeline_mode="artifact_pipeline",
        runtime_checkpoint_format="sharded",
        training_export_mode="flat",
        max_growth_steps_per_cycle=2,
        save_after_tree_growth_factor=1.0 if save else 1000.0,
        save_after_seconds=0.0 if save else 1e12,
    )


class RecordingRunner(AnemoneMorpionSearchRunner):
    """Observe real restores/steps without replacing scientific operations."""

    def __init__(
        self, *, cold: bool = False, eviction: bool = False, rollout: bool = False
    ) -> None:
        """Initialize real runner instrumentation."""
        super().__init__(
            AnemoneMorpionSearchRunnerArgs(
                search_args=default_search_args(
                    rollout=MorpionBootstrapRolloutConfig(
                        enabled=rollout, max_extra_steps=2
                    )
                ),
                runtime_checkpoint_format="sharded",
                growth_state_eviction_policy="frontier_cold" if eviction else "none",
                growth_state_eviction_recent_window=1,
                growth_state_eviction_scan_interval_steps=1,
                growth_state_eviction_payload_mode="delta_when_safe",
            )
        )
        self.cold = cold
        self.restores = 0
        self.runtime_ids: list[int] = []
        self.selected: list[object] = []

    def _load_runtime_from_checkpoint(self, *args: Any, **kwargs: Any) -> object:
        self.restores += 1
        assert self._runtime is None
        return super()._load_runtime_from_checkpoint(*args, **kwargs)

    def load_growth_runtime(self, *args: Any, **kwargs: Any) -> None:
        """Record graph identity and each real selected node."""
        if self.cold:
            self._warm_growth.invalidate()
        super().load_growth_runtime(*args, **kwargs)
        runtime = self.profile_runtime_root()
        self.runtime_ids.append(id(runtime))
        original = runtime.step

        def step() -> object:
            report = original()
            self.selected.append((report.selected_node_id, report.selected_depth))
            return report

        # Install once per actual graph, avoiding nested wrappers on warm calls.
        if not getattr(runtime, "_warm_test_recording", False):
            runtime.step = step
            runtime._warm_test_recording = True


def seed_workspace(path: Path) -> None:
    """Create a real persisted starting point, then discard its runner."""
    stages.run_pipeline_growth_stage(args_for(path), RecordingRunner())
    gc.collect()


def checkpoint(path: Path) -> dict[str, Any]:
    """Normalize serialization alone: no scientific fields are omitted."""
    paths = MorpionBootstrapPaths.from_work_dir(path)
    state = load_bootstrap_run_state(paths.run_state_path)
    root = paths.resolve_work_dir_path(state.latest_runtime_checkpoint_path)
    manifest = json.loads((root / "manifest.json").read_text())
    records: dict[str, Any] = {}
    for shard in manifest["shards"]:
        with zstandard.open(root / shard["path"], "rt") as stream:
            if shard["encoding"] == "jsonl_zst":
                records.setdefault(shard["kind"], []).extend(
                    json.loads(line) for line in stream
                )
            else:
                records[shard["kind"]] = json.load(stream)
    return records


def events(path: Path) -> list[dict[str, Any]]:
    """Read existing small structured observations in cycle order."""
    return [
        json.loads(f.read_text())
        for f in sorted(path.glob("pipeline/performance/*/runtime_load-*.json"))
    ]


@pytest.mark.parametrize("eviction", [False, True])
@pytest.mark.parametrize("rollout", [False, True])
def test_real_persisted_cycles_match_cold_reference(
    tmp_path: Path, eviction: bool, rollout: bool
) -> None:
    """Require complete parity, with cold fallback for unproven eviction cases."""
    warm_path, cold_path = tmp_path / "warm", tmp_path / "cold"
    seed_workspace(warm_path)
    seed_workspace(cold_path)
    warm = RecordingRunner(eviction=eviction, rollout=rollout)
    cold = RecordingRunner(cold=True, eviction=eviction, rollout=rollout)
    warm_state = stages.run_pipeline_growth_stage(
        replace(args_for(warm_path), tree_branch_limit=100000), warm, max_cycles=3
    )
    cold_state = stages.run_pipeline_growth_stage(
        replace(args_for(cold_path), tree_branch_limit=100000), cold, max_cycles=3
    )
    cold_required = eviction or rollout
    assert warm.restores == (3 if cold_required else 1)
    assert cold.restores == 3
    if not cold_required:
        assert len(set(warm.runtime_ids)) == 1
    assert warm.selected == cold.selected
    assert warm.current_tree_size() == cold.current_tree_size()
    assert warm.current_tree_branch_count() == cold.current_tree_branch_count()
    assert warm_state.latest_record_status == cold_state.latest_record_status
    assert warm_state.latest_frontier_status == cold_state.latest_frontier_status
    # Includes every node value/exactness/edge, selector state, RNG and evaluator version.
    assert checkpoint(warm_path) == checkpoint(cold_path)
    assert warm._random_generator.getstate() == cold._random_generator.getstate()
    assert [e["runtime_load_action"] for e in events(warm_path)][-3:] == [
        "cold_restore",
        "cold_restore_required" if cold_required else "warm_reuse",
        "cold_restore_required" if cold_required else "warm_reuse",
    ]


def test_no_save_discards_unsaved_growth_and_restart_restores(tmp_path: Path) -> None:
    """Do not silently change the existing loss of unsaved node/RNG mutations."""
    seed_workspace(tmp_path)
    initial = checkpoint(tmp_path)
    runner = RecordingRunner()
    state = stages.run_pipeline_growth_stage(
        args_for(tmp_path, save=False), runner, max_cycles=2
    )
    assert state.generation == 1
    assert runner.restores == 2
    assert runner.selected[:2] == runner.selected[2:]
    assert checkpoint(tmp_path) == initial
    restart = RecordingRunner()
    stages.run_pipeline_growth_stage(args_for(tmp_path), restart)
    assert restart.restores == 1


def test_eviction_owner_and_hooks_survive_warm_load(tmp_path: Path) -> None:
    """Keep cache, hot IDs, metrics and installed hooks attached to the same graph."""
    runner = RecordingRunner()
    stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    runtime = runner.profile_runtime_root()
    resolver = runner._live_compact_state_resolver
    metrics = runner._state_eviction_metrics
    recent = runner._growth_eviction_recent_node_ids
    before = runner.profile_state_eviction_runtime()
    original_step = runtime.step
    load_next(runner, tmp_path)
    assert runner.profile_runtime_root() is runtime
    assert runner._live_compact_state_resolver is resolver
    assert runner._state_eviction_metrics is metrics
    assert runner._growth_eviction_recent_node_ids is recent
    assert runtime.step is original_step
    assert runner.profile_state_eviction_runtime() == before


def load_next(runner: RecordingRunner, path: Path, **overrides: Any) -> None:
    """Exercise the explicit lifecycle using persisted current state."""
    paths = MorpionBootstrapPaths.from_work_dir(path)
    state = load_bootstrap_run_state(paths.run_state_path)
    kwargs = dict(
        tree_snapshot_path=paths.resolve_work_dir_path(
            state.latest_runtime_checkpoint_path
        ),
        model_bundle_path=None,
        effective_runtime_config=runner._warm_growth.loaded.runtime_config,
        paths=paths,
        run_state=state,
        force_evaluator=None,
        reevaluate_tree=False,
    )
    kwargs.update(overrides)
    runner.load_growth_runtime(**kwargs)


@pytest.mark.parametrize(
    "transition",
    [
        "force",
        "config",
        "patch",
        "cursor",
        "checkpoint",
        "rng",
        "branch_count",
        "nodes",
    ],
)
def test_incompatible_transition_requires_cold_restore(
    tmp_path: Path, transition: str
) -> None:
    """Conservatively disarm reuse for input or live-state drift."""
    runner = RecordingRunner()
    stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    overrides: dict[str, Any] = {}
    if transition == "force":
        overrides["force_evaluator"] = "other"
    elif transition == "config":
        overrides["effective_runtime_config"] = replace(
            runner._warm_growth.loaded.runtime_config, tree_branch_limit=12345
        )
    elif transition == "patch":
        paths.pipeline_reevaluation_patch_path.write_text("{}")
    elif transition == "cursor":
        paths.pipeline_reevaluation_cursor_path.write_text("{}")
    elif transition == "checkpoint":
        root = paths.resolve_work_dir_path(
            load_bootstrap_run_state(
                paths.run_state_path
            ).latest_runtime_checkpoint_path
        )
        manifest = root / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
    elif transition == "rng":
        runner._random_generator.random()
    elif transition == "branch_count":
        runner.profile_runtime_root().tree.branch_count += 1
    else:
        runner.grow(1)
    load_next(runner, tmp_path, **overrides)
    assert runner.restores == 1
    assert events(tmp_path)[-1]["runtime_load_action"] == "cold_restore_required"


def test_external_run_state_advance_refuses_stale_publication(tmp_path: Path) -> None:
    """A second publisher cannot silently be overwritten by a stale live worker."""
    runner = RecordingRunner()
    stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    data = json.loads(paths.run_state_path.read_text())
    data["generation"] += 1
    paths.run_state_path.write_text(json.dumps(data))
    with pytest.raises(GrowthRuntimeIdentityError, match="run_state changed"):
        load_next(runner, tmp_path)
    assert runner.restores == 0


@pytest.mark.parametrize("failure", ["checkpoint", "export", "manifest", "run_state"])
def test_interrupted_publication_never_arms_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Failed save/publication recovers from the previous committed generation."""
    seed_workspace(tmp_path)
    runner = RecordingRunner()
    target, name = {
        "checkpoint": (runner, "save_checkpoint"),
        "export": (stages, "_export_training_snapshot_for_generation"),
        "manifest": (stages, "_write_pipeline_manifest_for_generation"),
        "run_state": (stages, "save_bootstrap_run_state"),
    }[failure]
    original = getattr(target, name)
    monkeypatch.setattr(target, name, Mock(side_effect=OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    assert runner._warm_growth.committed is None
    monkeypatch.setattr(target, name, original)
    stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    assert runner.restores == 2


def test_model_identity_hashes_weights_at_same_path(tmp_path: Path) -> None:
    """A mutable bundle path cannot stand in for its actual code/weight identity."""
    bundle = tmp_path / "model"
    config = MorpionRegressorArgs(model_kind="linear")
    save_morpion_model_bundle(
        build_morpion_regressor(config), bundle, model_args=config
    )
    runner = RecordingRunner()
    work = tmp_path / "work"
    stages.run_pipeline_growth_stage(args_for(work), runner)
    load_next(runner, work, model_bundle_path=bundle)
    assert runner.restores == 1
    before = artifact_identity(bundle)
    save_morpion_model_bundle(
        build_morpion_regressor(config), bundle, model_args=config
    )
    assert artifact_identity(bundle) != before


def test_cold_path_releases_old_graph_before_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner-owned phase wrapper cycles must not keep the old tree alive."""
    runner = AnemoneMorpionSearchRunner()
    runner.load_or_create(None, None)
    old = weakref.ref(runner.profile_runtime_root())
    original = runner._create_fresh_runtime

    def create(*args: Any, **kwargs: Any) -> object:
        assert old() is None
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_create_fresh_runtime", create)
    runner.load_or_create(None, None)


def test_ram_guard_runs_before_warm_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live runtime never bypasses existing available-RAM admission checks."""
    runner = RecordingRunner()
    stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    guard = Mock(return_value=False)
    monkeypatch.setattr(stages, "log_available_ram_guard", guard)
    original = runner.profile_runtime_root()
    state = stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    assert state.generation == 1
    assert runner.profile_runtime_root() is original
    assert guard.call_args.kwargs["action"] == "checkpoint_load"
    assert len(events(tmp_path)) == 1


def test_unchanged_model_warm_parity_and_mutable_bundle_transition(
    tmp_path: Path,
) -> None:
    """Retain attachment/version semantics and detect new weights at an old path."""
    bundle = tmp_path / "bundle"
    config = MorpionRegressorArgs(model_kind="linear")
    save_morpion_model_bundle(
        build_morpion_regressor(config), bundle, model_args=config
    )
    runners = [RecordingRunner(), RecordingRunner(cold=True)]
    for name, runner in zip(["warm", "cold"], runners, strict=True):
        work = tmp_path / name
        paths = MorpionBootstrapPaths.from_work_dir(work)
        paths.ensure_directories()
        save_pipeline_active_model(
            MorpionPipelineActiveModel(
                generation=430,
                evaluator_name="default",
                model_bundle_path=str(bundle),
                updated_at_utc="fixture",
                source="external_seed",
                source_generation=430,
            ),
            paths.pipeline_active_model_path,
        )
        stages.run_pipeline_growth_stage(args_for(work), runner, max_cycles=3)
    assert checkpoint(tmp_path / "warm") == checkpoint(tmp_path / "cold")
    assert runners[0].restores == 0
    assert runners[1].restores == 2
    assert runners[0].selected == runners[1].selected
    save_morpion_model_bundle(
        build_morpion_regressor(config), bundle, model_args=config
    )
    stages.run_pipeline_growth_stage(args_for(tmp_path / "warm"), runners[0])
    assert runners[0].restores == 1
    assert events(tmp_path / "warm")[-1]["reason"] == "evaluator_identity_changed"


def test_pending_patch_applied_once_and_then_warm_parity(tmp_path: Path) -> None:
    """A consumed patch is not replayed and patched values survive later reuse."""
    runners = [RecordingRunner(), RecordingRunner(cold=True)]
    for name, runner in zip(["warm", "cold"], runners, strict=True):
        work = tmp_path / name
        stages.run_pipeline_growth_stage(args_for(work), runner)
        paths = MorpionBootstrapPaths.from_work_dir(work)
        patch = MorpionReevaluationPatch(
            patch_id="one-patch",
            created_at_utc="fixture",
            evaluator_generation=430,
            evaluator_name="default",
            model_bundle_path="external-model-430",
            tree_generation=1,
            rows=(MorpionReevaluationPatchRow(node_id="1", direct_value=3.25),),
        )
        save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)
        original = runner.apply_reevaluation_patch
        recorder = Mock(wraps=original)
        runner.apply_reevaluation_patch = recorder
        stages.run_pipeline_growth_stage(args_for(work), runner, max_cycles=3)
        assert recorder.call_count == 1
        assert not paths.pipeline_reevaluation_patch_path.exists()
    assert checkpoint(tmp_path / "warm") == checkpoint(tmp_path / "cold")
    assert events(tmp_path / "warm")[-3]["reason"] == "pending_reevaluation_patch"
    assert events(tmp_path / "warm")[-1]["runtime_load_action"] == "warm_reuse"


def test_publication_drift_during_growth_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check external advancement before publication as well as before load."""
    runner = RecordingRunner()
    stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    original = runner.grow

    def grow(steps: int) -> None:
        original(steps)
        data = json.loads(paths.run_state_path.read_text())
        data["generation"] = 99
        paths.run_state_path.write_text(json.dumps(data))

    monkeypatch.setattr(runner, "grow", grow)
    with pytest.raises(GrowthRuntimeIdentityError, match="run_state changed"):
        stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    assert load_bootstrap_run_state(paths.run_state_path).generation == 99


def test_incompatible_scientific_configuration_still_refuses(tmp_path: Path) -> None:
    """The warm optional path cannot evade persisted configuration validation."""
    runner = RecordingRunner()
    stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    persisted = json.loads(paths.bootstrap_config_path.read_text())
    persisted["experiment"]["variant"] = "incompatible-variant"
    paths.bootstrap_config_path.write_text(json.dumps(persisted))
    with pytest.raises(ValueError):
        stages.run_pipeline_growth_stage(args_for(tmp_path), runner)
