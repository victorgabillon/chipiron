"""Scientific values and source visibility across the versioned capsule boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    GenerationPublicationError,
    capsule_path,
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
)
from tests.environments.morpion_durable_v2_helpers import (
    ScientificPublicationRunner,
    publish_scientific_patch,
    run_scientific_publication,
    setup_scientific_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import (
    PATCHED,
)


def test_scientific_publication_retry_links_and_values(tmp_path: Path) -> None:
    """Commit exact alpha-0.5 effects once, current late parents and complete Values."""
    args, paths, initial = setup_scientific_baseline(tmp_path)
    publish_scientific_patch(paths)
    runner = ScientificPublicationRunner()
    runner.grow_delta = 1
    state = run_scientific_publication(args, runner)
    assert state.generation == 2 and runner.apply_calls == 1
    assert state.metadata[PUBLICATION_KEY]["schema_version"] == 2
    assert set(receipt_ledger(state)) == {"patch-1"}
    assert json.loads(
        (tmp_path / state.latest_runtime_checkpoint_path).read_bytes()
    ) == {**PATCHED, "new": 0.0}
    records = publication_records(state.metadata)
    old, new = (capsule_path(tmp_path, r["id"], 2) for r in records)
    assert (
        old / "tree_exports_sharded/generation_000001/records.jsonl"
    ).stat().st_ino == (
        new / "tree_exports_sharded/generation_000001/records.jsonl"
    ).stat().st_ino
    assert (old / "tree_exports_sharded/manifest.json").stat().st_ino != (
        new / "tree_exports_sharded/manifest.json"
    ).stat().st_ino
    source = tmp_path / state.latest_tree_snapshot_path
    reference = load_scientific_training_generation(source)
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
    ) as view:
        assert tuple(view.iter_nodes_in_export_order()) == reference.snapshot.nodes
        assert (
            dict(
                zip(
                    [n.node_id for n in reference.snapshot.nodes],
                    view.load_value_contracts([
                        n.node_id for n in reference.snapshot.nodes
                    ]),
                    strict=True,
                )
            )
            == reference.value_contracts
        )
        assert view.provenance == reference.provenance
        assert next(
            n for n in reference.snapshot.nodes if n.node_id == "leaf"
        ).parent_ids == ("root", "new")
        view.validate_source_identity()
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
    ) as view:
        assert view.index_reused
    restarted = ScientificPublicationRunner()
    recovered = run_scientific_publication(args, restarted, max_cycles=0)
    assert recovered == load_bootstrap_run_state(paths.run_state_path)
    assert recovered.generation == 2 and restarted.apply_calls == 0
    assert not paths.pipeline_reevaluation_patch_path.exists()
    validate_capsule(tmp_path, records[0])
    assert load_scientific_training_generation(
        tmp_path / initial.latest_tree_snapshot_path
    ).snapshot.nodes[1].parent_ids == ("root",)


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_managed_v2_dataset_reference_parity(tmp_path: Path, policy: str) -> None:
    """Compare full rows, confidence, metadata and JSONL; retain all four proof gates."""
    from chipiron.environments.morpion.bootstrap.cycle_dataset import (
        extract_rows_from_training_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.streaming_dataset import (
        IndexedDatasetSource,
        disk_dataset_context,
    )
    from chipiron.environments.morpion.learning import (
        save_morpion_supervised_rows_streaming,
    )

    args, _paths, state = setup_scientific_baseline(tmp_path)
    args = replace(
        args,
        dataset_confidence_policy="selected_value_v2",
        dataset_family_target_policy=policy,
    )
    source = tmp_path / state.latest_tree_snapshot_path
    ref = load_scientific_training_generation(source)
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
    ) as view:
        kwargs = dict(
            args=args,
            snapshot=ref.snapshot,
            generation=1,
            value_contracts=ref.value_contracts,
            source_provenance=ref.provenance,
        )
        if policy.startswith("pv_exact_then_"):
            with pytest.raises(ValueError):
                extract_rows_from_training_snapshot(**kwargs)
            with (
                pytest.raises(ValueError),
                disk_dataset_context(IndexedDatasetSource(view), args),
            ):
                pass
            return
        expected = extract_rows_from_training_snapshot(**kwargs)
        with disk_dataset_context(IndexedDatasetSource(view), args) as context:
            stream = context.stream(1)
            assert tuple(stream.rows) == expected.rows
            assert stream.metadata == expected.metadata
            save_morpion_supervised_rows_streaming(
                rows=context.rows(),
                metadata=stream.metadata,
                path=tmp_path / "indexed.jsonl",
            )
        save_morpion_supervised_rows_streaming(
            rows=expected.rows,
            metadata=expected.metadata,
            path=tmp_path / "reference.jsonl",
        )
        assert (tmp_path / "indexed.jsonl").read_bytes() == (
            tmp_path / "reference.jsonl"
        ).read_bytes()


def test_private_and_uncommitted_reader_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both readers refuse private/moved capsules until run state commits them."""
    import chipiron.environments.morpion.bootstrap.generation_publication as publication

    args, paths, baseline = setup_scientific_baseline(tmp_path)
    publish_scientific_patch(paths)
    actual = publication.publish_directory_if_absent
    seen = []

    def rejected(path: Path) -> None:
        with pytest.raises(GenerationPublicationError):
            load_scientific_training_generation(path)
        with pytest.raises(GenerationPublicationError):
            ShardedTrainingGenerationView(
                path, allow_scientific_v2=True, cache_dir=tmp_path / "consumer-cache"
            )
        assert not (tmp_path / "consumer-cache").exists()
        seen.append(path)

    def publish(staging: Path, final: Path, root: Path) -> None:
        if final.parent.name == "objects":
            relative = Path("tree_exports_sharded/generation_000002/generation.json")
            rejected(staging / relative)
            actual(staging, final, root)
            rejected(final / relative)
            assert load_bootstrap_run_state(paths.run_state_path).generation == 1
        else:
            actual(staging, final, root)

    with ShardedTrainingGenerationView(
        tmp_path / baseline.latest_tree_snapshot_path,
        allow_scientific_v2=True,
        cache_dir=tmp_path / "old-cache",
    ) as pinned:
        original = tuple(pinned.iter_nodes_in_export_order())
        with monkeypatch.context() as fault:
            fault.setattr(publication, "publish_directory_if_absent", publish)
            state = run_scientific_publication(args, ScientificPublicationRunner())
        assert tuple(pinned.iter_nodes_in_export_order()) == original
        pinned.validate_source_identity()
    assert len(seen) == 2
    assert load_scientific_training_generation(
        tmp_path / state.latest_tree_snapshot_path
    )
    assert not list(
        (tmp_path / "generation_publications_v2/objects").rglob("*.sqlite3")
    )


def test_managed_v2_hardlink_ctime_is_verified_by_sealed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linking a verified immutable descriptor between preparation/hash is legitimate."""
    import os

    _args, _paths, state = setup_scientific_baseline(tmp_path)
    source = tmp_path / state.latest_tree_snapshot_path
    actual = ShardedTrainingGenerationView._prepare_scientific_source

    def prepare(view: ShardedTrainingGenerationView) -> None:
        actual(view)
        os.link(source, tmp_path / "extra-link")

    with monkeypatch.context() as link:
        link.setattr(
            ShardedTrainingGenerationView, "_prepare_scientific_source", prepare
        )
        with ShardedTrainingGenerationView(
            source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
        ) as view:
            view.validate_source_identity()
    capsule = source.parents[2]
    with pytest.raises(GenerationPublicationError, match="immutable"):
        ShardedTrainingGenerationView(
            source, allow_scientific_v2=True, cache_dir=capsule / "cache"
        )
    assert not (capsule / "cache").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "generation_000001/records.jsonl",
        "generation_000001/updates.jsonl",
        "generation_000001/generation.json",
        "manifest.json",
    ],
)
def test_managed_v2_corruption_never_uses_valid_old_cache(
    tmp_path: Path, relative: str
) -> None:
    """A matching SQLite file cannot mask a modified committed source dependency."""
    _args, _paths, state = setup_scientific_baseline(tmp_path)
    source = tmp_path / state.latest_tree_snapshot_path
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
    ):
        pass
    path = source.parent.parent / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(GenerationPublicationError, match="changed"):
        ShardedTrainingGenerationView(
            source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
        )
    with pytest.raises(GenerationPublicationError, match="changed"):
        load_scientific_training_generation(source)


def test_explicit_v2_init_and_downgrade_rejections(tmp_path: Path) -> None:
    """No historical adoption, automatic activation, skipped export or producer change."""
    from chipiron.environments.morpion.bootstrap import run_pipeline_growth_stage
    from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
        initialize_scientific_v2_publication,
    )
    from tests.environments.morpion_durable_v2_helpers import scientific_producer
    from tests.environments.morpion_generation_publication_helpers import (
        setup_publication_baseline,
    )

    legacy_args, legacy_paths = setup_publication_baseline(tmp_path / "legacy")
    before = legacy_paths.run_state_path.read_bytes()
    with pytest.raises(ValueError, match="new isolated"):
        initialize_scientific_v2_publication(
            legacy_paths,
            load_bootstrap_run_state(legacy_paths.run_state_path),
            producer=scientific_producer(legacy_args),
        )
    assert legacy_paths.run_state_path.read_bytes() == before
    args, paths, _state = setup_scientific_baseline(tmp_path / "scientific")
    before = paths.run_state_path.read_bytes()
    for kwargs in (
        {},
        {"checkpoint_bound_receipts": True},
        {"checkpoint_bound_receipts": True, "durable_generation_publication": True},
    ):
        with pytest.raises(ValueError, match=r"requires|version"):
            run_pipeline_growth_stage(
                args, ScientificPublicationRunner(), max_cycles=0, **kwargs
            )
    with pytest.raises(ValueError, match="every saved"):
        run_scientific_publication(
            replace(args, growth_skip_training_export=True),
            ScientificPublicationRunner(),
        )
    with pytest.raises(ValueError, match="producer"):
        run_pipeline_growth_stage(
            args,
            ScientificPublicationRunner(),
            max_cycles=0,
            checkpoint_bound_receipts=True,
            durable_generation_publication=True,
            scientific_v2_producer={**scientific_producer(args), "revision": "changed"},
        )
    assert paths.run_state_path.read_bytes() == before


def test_v2_patch_requires_exact_committed_scientific_identity(tmp_path: Path) -> None:
    """A generation number alone is insufficient provenance in the v2 transaction."""
    from tests.environments.morpion_patch_receipt_helpers import publish_fixture_patch

    args, paths, _state = setup_scientific_baseline(tmp_path)
    publish_fixture_patch(paths)
    before = paths.run_state_path.read_bytes()
    runner = ScientificPublicationRunner()
    with pytest.raises(ValueError, match="scientific publication identity"):
        run_scientific_publication(args, runner)
    assert runner.apply_calls == 0 and paths.run_state_path.read_bytes() == before
    assert paths.pipeline_reevaluation_patch_path.exists()


def test_publication_validation_does_not_use_full_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sealing a v2 generation may use only the existing bounded validation index."""
    import chipiron.environments.morpion.bootstrap.sharded_training_v2 as scientific

    def forbidden(*_args: object, **_kwargs: object) -> None:
        message = "Full scientific snapshot invoked by publisher"
        raise AssertionError(message)

    monkeypatch.setattr(scientific, "load_scientific_training_generation", forbidden)
    monkeypatch.setattr(scientific, "_load_scientific_training_generation", forbidden)
    args, paths, _state = setup_scientific_baseline(tmp_path)
    publish_scientific_patch(paths)
    assert (
        run_scientific_publication(args, ScientificPublicationRunner()).generation == 2
    )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("cursor", [None, "leaf", "zz-wrap"])
def test_managed_v2_reevaluation_complete_patch_and_cursor_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool, cursor: str | None
) -> None:
    """Full-reference and indexed patch bytes retain the same sealed source binding."""
    import chipiron.environments.morpion.bootstrap.reevaluation_worker as worker
    from chipiron.environments.morpion.bootstrap import (
        MorpionReevaluationCursor,
        save_reevaluation_cursor,
    )
    from tests.environments.test_morpion_scientific_training_reader import StateScore
    from tests.environments.test_morpion_streaming_reevaluation import run_once

    _args, paths, state = setup_scientific_baseline(tmp_path)
    oracle = load_scientific_training_generation(
        tmp_path / state.latest_tree_snapshot_path
    )
    save_reevaluation_cursor(
        MorpionReevaluationCursor(
            evaluator_generation=1,
            evaluator_name="default",
            model_bundle_path="models/generation_000001/default",
            next_node_cursor=cursor,
            tree_generation=1,
            updated_at_utc="2026-10-21T12:00:00Z",
            completed_full_pass_count=2,
        ),
        paths.pipeline_reevaluation_cursor_path,
    )
    initial_cursor = paths.pipeline_reevaluation_cursor_path.read_bytes()

    def evaluator(**_kwargs: object) -> object:
        result = worker.MorpionActiveModelNodeReevaluationEvaluator(
            tmp_path / "common-model"
        )
        result._evaluator = StateScore()
        return result

    monkeypatch.setattr(worker, "build_active_model_reevaluation_evaluator", evaluator)
    monkeypatch.setattr(
        worker, "load_reevaluation_training_tree_snapshot", lambda _: oracle.snapshot
    )
    expected = run_once(paths, streaming=False, use_snapshot_value_fallback=fallback)
    patch = paths.pipeline_reevaluation_patch_path.read_bytes()
    final_cursor = paths.pipeline_reevaluation_cursor_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.unlink()
    paths.pipeline_reevaluation_cursor_path.write_bytes(initial_cursor)
    monkeypatch.setattr(
        worker,
        "load_reevaluation_training_tree_snapshot",
        lambda _: pytest.fail("Full oracle called by bounded production path"),
    )
    actual = run_once(
        paths, allow_scientific_v2=True, use_snapshot_value_fallback=fallback
    )
    assert actual == expected
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == patch
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == final_cursor
    binding = json.loads(patch)["metadata"]["scientific_publication"]
    assert (
        binding["scientific_identity"]
        == publication_records(state.metadata)[0]["scientific_identity"]
    )


@pytest.mark.parametrize("suffix", ["json.zst", "sharded"])
def test_real_runner_checkpoint_selector_and_scientific_parity(
    tmp_path: Path, suffix: str
) -> None:
    """A capsule preserves actual live Values, selector counters and RNG continuation."""
    from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs

    from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
    from chipiron.environments.morpion.bootstrap.config import (
        bootstrap_config_from_args,
        bootstrap_config_sha256,
        bootstrap_config_to_dict,
    )
    from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
    from chipiron.environments.morpion.bootstrap.run_state import (
        initialize_bootstrap_run_state,
    )
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        AnemoneMorpionSearchRunner,
        AnemoneMorpionSearchRunnerArgs,
        default_search_args,
    )
    from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
        initialize_scientific_v2_publication,
    )
    from tests.environments.morpion_durable_v2_helpers import (
        scientific_args,
        scientific_producer,
    )
    from tests.environments.test_morpion_checkpoint_parity import _selection_signature

    args = replace(
        scientific_args(tmp_path),
        runtime_checkpoint_format="sharded" if suffix == "sharded" else "json-zst",
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    config = bootstrap_config_from_args(args)
    atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
    runner_args = AnemoneMorpionSearchRunnerArgs(
        search_args=replace(
            default_search_args(),
            stopping_criterion=TreeBranchLimitArgs(
                type="tree_branch_limit", tree_branch_limit=4096
            ),
        ),
        random_seed=4,
        runtime_checkpoint_format=args.runtime_checkpoint_format,
    )
    continuous = AnemoneMorpionSearchRunner(runner_args)
    continuous.load_or_create(None, None)
    continuous.grow(1)
    checkpoint = tmp_path / f"genesis-checkpoint.{suffix}"
    continuous.save_checkpoint(checkpoint)
    source = continuous.export_scientific_training_snapshot(
        tmp_path / "genesis-source", generation=1, producer=scientific_producer(args)
    )
    before = load_scientific_training_generation(source)
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
        latest_tree_snapshot_path=paths.relative_to_work_dir(source),
        last_save_unix_s=1e9,
        metadata={"bootstrap_config_hash": bootstrap_config_sha256(config)},
    )
    committed = initialize_scientific_v2_publication(
        paths, state, producer=scientific_producer(args)
    )
    after = load_scientific_training_generation(
        tmp_path / committed.latest_tree_snapshot_path
    )
    assert after == before
    restored = AnemoneMorpionSearchRunner(replace(runner_args, random_seed=999))
    restored.load_or_create(tmp_path / committed.latest_runtime_checkpoint_path, None)
    assert (
        restored._random_generator.getstate() == continuous._random_generator.getstate()
    )
    restored_source = restored.export_scientific_training_snapshot(
        tmp_path / "restored-source", generation=1, producer=scientific_producer(args)
    )
    assert load_scientific_training_generation(restored_source) == before
    for _ in range(2):
        continuous.grow(1)
        restored.grow(1)
        assert _selection_signature(restored) == _selection_signature(continuous)
        assert (
            restored._random_generator.getstate()
            == continuous._random_generator.getstate()
        )
        assert restored.current_tree_size() == continuous.current_tree_size()


@pytest.fixture(scope="module")
def committed_endgame(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, object, object]:
    """Publish the actual reachable FORCED(35)/neural ESTIMATE(42) checkpoint."""
    from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
    from chipiron.environments.morpion.bootstrap.config import (
        bootstrap_config_from_args,
        bootstrap_config_sha256,
        bootstrap_config_to_dict,
    )
    from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
    from chipiron.environments.morpion.bootstrap.run_state import (
        initialize_bootstrap_run_state,
    )
    from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
        initialize_scientific_v2_publication,
    )
    from tests.environments.morpion_durable_v2_helpers import (
        scientific_args,
        scientific_producer,
    )
    from tests.environments.test_morpion_value_certainty import (
        endgame,
        make_runner,
        model_patch,
    )

    root = tmp_path_factory.mktemp("durable-actual-endgame")
    args = scientific_args(root)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    config = bootstrap_config_from_args(args)
    atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
    with pytest.MonkeyPatch.context() as patch:
        late, _terminal = endgame.__wrapped__()
        runner = make_runner(patch, state=late)
        runner.apply_reevaluation_patch(model_patch(runner, patch, 42, root))
        checkpoint = root / "genesis.json.zst"
        runner.save_checkpoint(checkpoint)
        source = runner.export_scientific_training_snapshot(
            root / "genesis-scientific",
            generation=1,
            producer=scientific_producer(args),
        )
    before = load_scientific_training_generation(source)
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
        latest_tree_snapshot_path=paths.relative_to_work_dir(source),
        last_save_unix_s=1e9,
        metadata={"bootstrap_config_hash": bootstrap_config_sha256(config)},
    )
    state = initialize_scientific_v2_publication(
        paths, state, producer=scientific_producer(args)
    )
    source = root / state.latest_tree_snapshot_path
    assert load_scientific_training_generation(source) == before
    return source, args, before


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("backup", [False, True])
@pytest.mark.parametrize("exact_only", [False, True])
def test_managed_real_endgame_confidence_all_policies(
    tmp_path: Path,
    committed_endgame: tuple,
    policy: str,
    backup: bool,
    exact_only: bool,
) -> None:
    """Retain every selected-source certainty assertion through the durable boundary."""
    from chipiron.environments.morpion.bootstrap.cycle_dataset import (
        extract_rows_from_training_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.streaming_dataset import (
        IndexedDatasetSource,
        disk_dataset_context,
    )
    from chipiron.environments.morpion.learning import (
        save_morpion_supervised_rows_streaming,
    )

    source, args, oracle = committed_endgame
    args = replace(
        args,
        dataset_confidence_policy="selected_value_v2",
        dataset_family_target_policy=policy,
        use_backed_up_value=backup,
        require_exact_or_terminal=exact_only,
    )
    assert oracle.value_contracts["0"]["certainties"] == {
        "direct": "ESTIMATE",
        "tree": "FORCED",
        "effective": "FORCED",
        "target": "FORCED",
    }
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
    ) as view:
        if policy.startswith("pv_exact_then_"):
            with pytest.raises(ValueError, match="gates pv_exact_then"):
                extract_rows_from_training_snapshot(
                    args=args,
                    snapshot=oracle.snapshot,
                    generation=1,
                    value_contracts=oracle.value_contracts,
                    source_provenance=oracle.provenance,
                )
            with (
                pytest.raises(ValueError, match="gates pv_exact_then"),
                disk_dataset_context(IndexedDatasetSource(view), args),
            ):
                pass
            return
        expected = extract_rows_from_training_snapshot(
            args=args,
            snapshot=oracle.snapshot,
            generation=1,
            value_contracts=oracle.value_contracts,
            source_provenance=oracle.provenance,
        )
        with disk_dataset_context(IndexedDatasetSource(view), args) as context:
            stream = context.stream(1)
            assert tuple(stream.rows) == expected.rows
            assert stream.metadata == expected.metadata
            save_morpion_supervised_rows_streaming(
                rows=context.rows(),
                metadata=stream.metadata,
                path=tmp_path / "indexed.jsonl",
            )
        save_morpion_supervised_rows_streaming(
            rows=expected.rows,
            metadata=expected.metadata,
            path=tmp_path / "reference.jsonl",
        )
        assert (tmp_path / "indexed.jsonl").read_bytes() == (
            tmp_path / "reference.jsonl"
        ).read_bytes()
    assert len(expected.rows) == (1 if exact_only and not backup else 2)
    terminal = expected.rows[-1]
    assert terminal.target_value == 35 and terminal.is_terminal and terminal.is_exact
    assert (
        terminal.metadata["target_confidence"]["training_target_certainty"]
        == "TERMINAL"
    )
    if not exact_only or backup:
        parent = expected.rows[0]
        confidence = parent.metadata["target_confidence"]
        assert confidence["node_is_exact"] is True
        assert confidence["raw_target_value"] == (35 if backup else 42)
        assert confidence["raw_target_certainty"] == (
            "FORCED" if backup else "ESTIMATE"
        )
        assert confidence["training_target_certainty"] == (
            "FORCED" if backup else "ESTIMATE"
        )
        assert parent.is_exact is backup
        assert parent.target_value == (
            35
            if backup
            else {
                "none": 42,
                "pv_mean_prediction": 38.5,
                "pv_min_prediction": 35,
                "pv_blend_mean_prediction": 41.125,
                "pv_blend_min_prediction": 40.25,
            }[policy]
        )


@pytest.mark.parametrize(
    "boundary",
    ["before_capsule", "after_capsule", "before_state", "after_state", "discovery"],
)
def test_genesis_failure_retry_uses_only_prepared_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Genesis retries fresh inputs; after replacement recovery uses committed state."""
    import os

    import chipiron.environments.morpion.bootstrap.generation_publication as publication
    from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
        GrowthPatchReceiptSession,
    )
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        artifact_digest,
    )
    from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
        initialize_scientific_v2_publication,
    )
    from tests.environments.morpion_durable_v2_helpers import scientific_producer

    args, paths, prepared = setup_scientific_baseline(tmp_path, initialize=False)
    input_digest = artifact_digest(tmp_path / "genesis-scientific-v2")
    actual_publish, actual_replace, actual_manifest = (
        publication.publish_directory_if_absent,
        os.replace,
        GrowthPatchReceiptSession.publish_manifest,
    )

    def fail() -> None:
        message = "injected genesis interruption"
        raise OSError(message)

    def publish(staging: Path, final: Path, root: Path) -> None:
        if boundary == "before_capsule":
            fail()
        actual_publish(staging, final, root)
        if boundary == "after_capsule":
            fail()

    def replace_state(source: str | Path, destination: str | Path) -> None:
        is_state = Path(destination) == paths.run_state_path
        if is_state and boundary == "before_state":
            fail()
        actual_replace(source, destination)
        if is_state and boundary == "after_state":
            fail()

    def manifest(session: GrowthPatchReceiptSession, state: object) -> None:
        if boundary == "discovery":
            fail()
        actual_manifest(session, state)

    with monkeypatch.context() as fault:
        fault.setattr(publication, "publish_directory_if_absent", publish)
        fault.setattr(os, "replace", replace_state)
        fault.setattr(GrowthPatchReceiptSession, "publish_manifest", manifest)
        with pytest.raises(OSError, match="genesis interruption"):
            initialize_scientific_v2_publication(
                paths, prepared, producer=scientific_producer(args)
            )
    if paths.run_state_path.exists():
        recovered = run_scientific_publication(
            args, ScientificPublicationRunner(), max_cycles=0
        )
    else:
        for path in (tmp_path / "generation_publications_v2/objects").glob(
            "*/tree_exports_sharded/generation_000001/generation.json"
        ):
            with pytest.raises((ValueError, FileNotFoundError)):
                load_scientific_training_generation(path)
        recovered = initialize_scientific_v2_publication(
            paths, prepared, producer=scientific_producer(args)
        )
    assert recovered.generation == 1
    assert len(publication_records(recovered.metadata)) == 1
    assert artifact_digest(tmp_path / "genesis-scientific-v2") == input_digest
    assert len(list((tmp_path / "generation_publications_v2/objects").iterdir())) == 1
    assert not list((tmp_path / "generation_publications_v2/staging").iterdir())
