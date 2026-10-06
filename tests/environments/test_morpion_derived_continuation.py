"""Opt-in derivation, sequential barriers and checkpoint-only drain regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import (
    MorpionBootstrapRolloutConfig,
    MorpionBootstrapSearchConfig,
    bootstrap_config_from_args,
    bootstrap_config_from_dict,
    save_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.derived.orchestrator import (
    STAGES,
    PipelineGenerationStages,
    run_sequential,
)
from chipiron.environments.morpion.bootstrap.derived.prepare import (
    _checkpoint_metadata,
    inspect_source,
    migrate_config,
    prepare_workspace,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import (
    DerivationError,
    atomic_json,
    read_json,
    sha256,
)
from chipiron.environments.morpion.bootstrap.derived.runtime import make_runner
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.pipeline.cursors import (
    active_model_generation_for_training_guard,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineActiveModel,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    MorpionBootstrapRunState,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.legacy_graph.config import (
    LegacyGraphConfig,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressorArgs,
    build_morpion_regressor,
)


@pytest.fixture
def historical(tmp_path: Path) -> Path:
    """Build a tiny real checkpoint with missing RNG, legacy config and external model 430."""
    root = tmp_path / "history"
    root.mkdir()
    legacy = LegacyGraphConfig(
        graph_max_tokens=16,
        graph_d_model=8,
        graph_n_head=2,
        graph_n_layer=1,
        graph_dim_feedforward=16,
    )
    specs = {
        "linear_41": MorpionEvaluatorSpec(
            name="linear_41",
            model_type="linear",
            hidden_sizes=None,
            num_epochs=1,
            batch_size=16,
            learning_rate=0.001,
        ),
        "transformer": MorpionEvaluatorSpec(
            name="transformer",
            model_type="entity_token_transformer_value_net",
            hidden_sizes=None,
            num_epochs=1,
            batch_size=16,
            learning_rate=0.001,
            legacy_graph_tokens=legacy,
        ),
    }
    args = MorpionBootstrapArgs(
        work_dir=root,
        pipeline_mode="artifact_pipeline",
        training_export_mode="sharded",
        runtime_checkpoint_format="sharded",
        max_growth_steps_per_cycle=10,
        tree_branch_limit=1000,
        search=MorpionBootstrapSearchConfig(
            rollout=MorpionBootstrapRolloutConfig(enabled=True, max_extra_steps=1)
        ),
        training_device="cpu",
        evaluators_config=MorpionEvaluatorsConfig(specs),
    )
    config = bootstrap_config_from_args(args)
    save_bootstrap_config(config, root / "bootstrap_config.json")
    raw = read_json(root / "bootstrap_config.json")
    item = raw["evaluators"]["evaluators"]["transformer"]
    block = item.pop("legacy_graph_tokens")
    block.pop("representation")
    item.update(block)
    atomic_json(root / "bootstrap_config.json", raw)
    model_args = MorpionRegressorArgs()
    model_dir = root / "models/generation_000430/linear_41"
    save_morpion_model_bundle(
        build_morpion_regressor(model_args), model_dir, model_args=model_args
    )
    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=430,
            evaluator_name="linear_41",
            model_bundle_path="models/generation_000430/linear_41",
            updated_at_utc="2026-10-05T00:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    runner = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            runtime_checkpoint_format="sharded",
            search_args=default_search_args(rollout=args.search.rollout),
        )
    )
    runner.load_or_create(None, model_dir)
    runner.grow(1)
    checkpoint = root / "search_checkpoints/generation_000038.sharded"
    runner.save_checkpoint(checkpoint)
    export = runner.export_sharded_training_tree_snapshot(
        root / "tree_exports_sharded", generation=38
    )
    state = MorpionBootstrapRunState(
        generation=38,
        cycle_index=42,
        latest_tree_snapshot_path=str(export.relative_to(root)),
        latest_runtime_checkpoint_path=str(checkpoint.relative_to(root)),
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name="linear_41",
        tree_size_at_last_save=runner.current_tree_size(),
        last_save_unix_s=0,
    )
    save_bootstrap_run_state(state, root / "run_state.json")
    metadata = _checkpoint_metadata(checkpoint)
    metadata["rng_state"] = None
    metadata.pop("rollout_rng_state", None)
    (checkpoint / "metadata.json.zst").write_bytes(
        zstandard.ZstdCompressor().compress(json.dumps(metadata).encode())
    )
    return root


def _plan(source: Path, target: Path, *, count: int = 2) -> dict:
    result = inspect_source(
        source,
        target,
        generation=38,
        expected_nodes=read_json(source / "run_state.json")["tree_size_at_last_save"],
        growth_steps=2,
        max_generations=count,
        search_seed=123,
        rollout_seed=0,
        training_seed=7,
        code_sha="test-sha",
    )
    result["code_root"] = str(Path(__file__).parents[2])
    return result


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}


def test_prepare_preserves_history_and_resets_independent_rng_deterministically(
    historical: Path, tmp_path: Path
) -> None:
    """Verify prepare preserves history and resets independent rng deterministically."""
    before = _hashes(historical)
    first = _plan(historical, tmp_path / "a")
    assert not (tmp_path / "a").exists()  # inspection is read-only
    prepare_workspace(first)
    second = _plan(historical, tmp_path / "b")
    prepare_workspace(second)
    assert _hashes(historical) == before
    a = _checkpoint_metadata(tmp_path / "a" / first["source_checkpoint"])
    b = _checkpoint_metadata(tmp_path / "b" / second["source_checkpoint"])
    assert a == b
    assert a["rng_state"] != a["rollout_rng_state"]
    assert first["experiment_id"] == second["experiment_id"]
    for rel, old in before.items():
        if rel.startswith(first["source_checkpoint"]) and not rel.endswith((
            "metadata.json.zst",
            "manifest.json",
        )):
            assert (
                sha256(tmp_path / "a" / rel) == old
            )  # selector/latest expansions/payloads unchanged
    with pytest.raises(DerivationError, match="existing target"):
        _plan(historical, tmp_path / "a")
    source_args = read_json(historical / "bootstrap_config.json")
    effective, mapping = migrate_config(source_args, growth_steps=2)
    assert len(mapping) == 9
    effective["runtime"]["max_growth_steps_per_cycle"] = 10
    spec = effective["evaluators"]["evaluators"]["transformer"]
    block = spec.pop("legacy_graph_tokens")
    block.pop("representation")
    spec.update(block)
    assert effective == source_args


@pytest.mark.parametrize(
    "mutation", ["generation", "node_count", "checkpoint", "unknown_graph_field"]
)
def test_prepare_refuses_ambiguous_source(
    historical: Path, tmp_path: Path, mutation: str
) -> None:
    """Verify prepare refuses ambiguous source."""
    state = read_json(historical / "run_state.json")
    if mutation == "generation":
        state["generation"] = 37
    elif mutation == "node_count":
        p = historical / state["latest_tree_snapshot_path"]
        x = read_json(p)
        x["node_count"] += 1
        atomic_json(p, x)
    elif mutation == "checkpoint":
        state["latest_runtime_checkpoint_path"] = state["latest_tree_snapshot_path"]
    else:
        p = historical / "bootstrap_config.json"
        x = read_json(p)
        x["evaluators"]["evaluators"]["transformer"]["graph_unknown"] = 1
        atomic_json(p, x)
    atomic_json(historical / "run_state.json", state)
    with pytest.raises(DerivationError):
        _plan(historical, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_external_publication_is_derived_only(historical: Path, tmp_path: Path) -> None:
    """Verify external publication is derived only."""
    assert (
        active_model_generation_for_training_guard(
            MorpionBootstrapPaths.from_work_dir(historical)
        )
        == 430
    )
    target = prepare_workspace(_plan(historical, tmp_path / "derived"))
    assert (
        active_model_generation_for_training_guard(
            MorpionBootstrapPaths.from_work_dir(target)
        )
        == 38
    )
    active = read_json(target / "pipeline/active_model.json")
    assert active["generation"] == active["source_generation"] == 430
    assert active["local_trained_generation"] is None


@pytest.mark.parametrize("failed_stage", STAGES)
def test_ten_generation_barriers_resume_without_skipping(
    historical: Path, tmp_path: Path, failed_stage: str
) -> None:
    """Verify ten generation barriers resume without skipping."""
    target = prepare_workspace(_plan(historical, tmp_path / "derived", count=10))
    calls = []

    class FakeStages:
        fail = True

        def run_stage(self, stage: str, generation: int) -> dict[str, int]:
            if self.fail and generation == 40 and stage == failed_stage:
                self.fail = False
                raise RuntimeError("interrupted")
            calls.append((generation, stage))
            return {"generation": generation}

    backend = FakeStages()
    with pytest.raises(RuntimeError, match="interrupted"):
        run_sequential(target, backend)
    assert (
        read_json(target / "derived_orchestration.json")["completed_generation"] == 39
    )
    journal = run_sequential(target, backend)
    assert journal["completed_generation"] == 48
    assert calls == [
        (generation, stage) for generation in range(39, 49) for stage in STAGES
    ]
    run_sequential(target, backend)
    assert len(calls) == 50


def test_strict_restore_never_creates_fresh_tree(
    historical: Path, tmp_path: Path
) -> None:
    """Verify strict restore never creates fresh tree."""
    target = prepare_workspace(_plan(historical, tmp_path / "derived"))
    config = bootstrap_config_from_dict(read_json(target / "bootstrap_config.json"))
    runner = make_runner(target, config, 123)
    with pytest.raises(DerivationError, match="fresh initialization"):
        runner.load_or_create(None, None)
    with pytest.raises(DerivationError, match="fallback"):
        runner.load_or_create(
            target / "tree_exports_sharded/generation_000038.json", None
        )


@pytest.mark.parametrize("interruption", ["before_commit", "after_commit"])
def test_real_tiny_two_generation_pipeline_promotes_and_drains(
    historical: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    """Verify real tiny two generation pipeline promotes and drains."""
    before = _hashes(historical)
    target = prepare_workspace(_plan(historical, tmp_path / "derived", count=2))
    from chipiron.environments.morpion.bootstrap.derived import runtime as drain_runtime

    original_commit = drain_runtime.atomic_json
    original_ack = drain_runtime.delete_reevaluation_patch
    failed = False

    def commit(path: Path, data: dict) -> None:
        nonlocal failed
        if interruption == "before_commit" and not failed:
            failed = True
            message = "simulated interruption before pointer commit"
            raise OSError(message)
        original_commit(path, data)

    def acknowledge(path: Path) -> None:
        nonlocal failed
        if interruption == "after_commit" and not failed:
            failed = True
            message = "simulated interruption after pointer commit"
            raise OSError(message)
        original_ack(path)

    monkeypatch.setattr(drain_runtime, "atomic_json", commit)
    monkeypatch.setattr(drain_runtime, "delete_reevaluation_patch", acknowledge)
    with pytest.raises(OSError, match="simulated interruption"):
        run_sequential(target, PipelineGenerationStages(target))
    assert (
        read_json(target / "derived_orchestration.json")["completed_generation"] == 38
    )
    assert MorpionBootstrapPaths.from_work_dir(
        target
    ).pipeline_reevaluation_patch_path.exists()
    result = run_sequential(target, PipelineGenerationStages(target))
    assert result["completed_generation"] == 40
    state = read_json(target / "run_state.json")
    assert state["generation"] == 40
    assert state["cycle_index"] == 44
    active = read_json(target / "pipeline/active_model.json")
    assert active["generation"] == 40 and active["source"] == "local_training"
    assert state["metadata"]["derived_drain"]["growth_steps"] == 0
    assert state["metadata"]["derived_drain"]["full_pass"]
    assert not MorpionBootstrapPaths.from_work_dir(
        target
    ).pipeline_reevaluation_patch_path.exists()
    for generation in (39, 40):
        evidence = result["generations"][str(generation)]["evidence"]
        assert evidence["training"]["evaluators"] == ["linear_41", "transformer"]
        assert evidence["drain"]["rows"] == evidence["growth"]["nodes"]
        assert evidence["growth"]["growth_steps"] == 2
        observations = [
            read_json(path)
            for path in (
                target / "pipeline/performance" / f"generation_{generation:06d}"
            ).glob("*.json")
        ]
        assert {
            "growth",
            "checkpoint",
            "export",
            "dataset",
            "training",
            "reevaluation",
            "patch_apply",
        } <= {item["stage"] for item in observations}
        growth = next(item for item in observations if item["stage"] == "growth")
        assert growth["nodes_added"] == evidence["growth"]["nodes_added"]
        assert growth["growth_steps"] == 2
    assert _hashes(historical) == before
    assert not (target / "search_checkpoints/generation_000041.sharded").exists()


@pytest.mark.parametrize("mode", ["--dry-run", "--prepare-only"])
def test_cli_preparation_never_enters_execution(
    historical: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Both preparation modes avoid stage construction, and launch needs confirmation."""
    from chipiron.environments.morpion.bootstrap.derived import cli

    identity = {
        "code_root": str(tmp_path),
        "code_sha": "validated-test-sha",
        "package_fingerprint": "test-fingerprint",
    }
    monkeypatch.setattr(cli, "code_identity", lambda _: identity)
    monkeypatch.setattr(cli, "environment_identity", lambda: {"test": "environment"})

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Preparation or unconfirmed launch entered a scientific stage")

    monkeypatch.setattr(PipelineGenerationStages, "__init__", forbidden)
    target = tmp_path / "derived"
    assert (
        cli.main([
            "derive",
            "--source-work-dir",
            str(historical),
            "--target-work-dir",
            str(target),
            "--code-root",
            str(tmp_path),
            "--checkpoint-generation",
            "38",
            "--expected-node-count",
            str(read_json(historical / "run_state.json")["tree_size_at_last_save"]),
            "--reset-rng",
            "--search-seed",
            "0",
            "--rollout-seed",
            "0",
            "--training-seed",
            "0",
            "--enable-legacy-graph-tokens-v1",
            mode,
        ])
        == 0
    )
    if mode == "--dry-run":
        assert not target.exists()
        return
    before = _hashes(target)
    assert cli.main(["run", "--work-dir", str(target), "--dry-run"]) == 0
    assert _hashes(target) == before
    assert cli.main(["run", "--work-dir", str(target), "--launch"]) == 2
    assert _hashes(target) == before
    assert not (target / ".derived_execution.lock").exists()


@pytest.mark.parametrize(
    "mutation", ["config", "checkpoint", "environment", "code", "incomplete"]
)
def test_prepared_workspace_integrity_gates(
    historical: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Reject changed scientific inputs before any runner is constructed."""
    from chipiron.environments.morpion.bootstrap.derived import cli

    plan = _plan(historical, tmp_path / "derived")
    identity = {key: plan[key] for key in ("code_root", "code_sha")}
    identity["package_fingerprint"] = "test-fingerprint"
    plan.update(identity)
    plan["environment"] = {"test": "environment"}
    target = prepare_workspace(plan)
    monkeypatch.setattr(cli, "code_identity", lambda _: identity)
    monkeypatch.setattr(cli, "environment_identity", lambda: {"test": "environment"})
    assert cli.validate_prepared(target)["current_generation"] == 38
    if mutation == "config":
        (target / "bootstrap_config.json").write_text("{}")
    elif mutation == "checkpoint":
        (target / plan["source_checkpoint"] / "metadata.json.zst").write_bytes(
            b"changed"
        )
    elif mutation == "environment":
        monkeypatch.setattr(cli, "environment_identity", lambda: {"test": "changed"})
    elif mutation == "code":
        identity["code_sha"] = "changed"
    else:
        (target / "preparation_incomplete.json").write_text("{}")
    with pytest.raises(DerivationError):
        cli.validate_prepared(target)
