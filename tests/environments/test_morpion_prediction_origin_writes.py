"""Actual native writes and capsule atomicity for the explicit C3 tracking mode."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import run_pipeline_growth_stage
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    committed_evaluator,
    publication_records,
    read_object,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.prediction_origin_publication import (
    read_committed_prediction_origins,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
)
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard
from tests.environments.test_morpion_model_activation_edges import propose
from tests.environments.test_morpion_value_certainty import endgame as endgame

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.fast


def native(runner: AnemoneMorpionSearchRunner) -> Any:
    """Inspect the actual native tree in focused boundary assertions."""
    assert runner._runtime is not None
    return runner._runtime


def tracked(paths: Any) -> Any:
    """Read only committed evidence, using the production reader."""
    state = load_bootstrap_run_state(paths.run_state_path)
    result = read_committed_prediction_origins(
        paths.work_dir, publication_records(state.metadata)[-1]
    )
    assert result is not None
    return result


def grow(args: Any, paths: Any, *, max_cycles: int = 1) -> Any:
    """Use the existing Growth stage and its ordinary publication/receipt session."""
    before = load_bootstrap_run_state(paths.run_state_path)
    runner, _ = _runner(paths, before)
    producer = publication_records(before.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=max_cycles,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )
    return load_bootstrap_run_state(paths.run_state_path)


@pytest.mark.parametrize("layout", ["sharded", "json-zst"])
def test_actual_genesis_growth_patch_and_full_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    """Real native 0.4.1 writes, exact receipt evidence, restore and existing activation."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, initial = setup_activation(
        tmp_path / "work", layout=layout, prediction_origin_tracking=True
    )
    first = tracked(paths)
    assert first.current_origins and {
        origin.kind for origin in first.current_origins.values()
    } == {"genesis"}
    assert {origin.evaluator_sha256 for origin in first.current_origins.values()} == {
        artifact_digest(paths.work_dir / "seed")
    }
    assert propose(args).patch_written
    patch_bytes = paths.pipeline_reevaluation_patch_path.read_bytes()
    grown = grow(args, paths)
    second = tracked(paths)
    assert grown.generation == initial.generation + 1
    assert len(second.creation_generations) > len(first.creation_generations)
    assert {origin.kind for origin in second.current_origins.values()} >= {
        "genesis",
        "growth_birth",
        "accepted_patch",
    }
    receipt = next(iter(receipt_ledger(grown).values()))
    head = publication_records(grown.metadata)[-1]
    root = (paths.work_dir / head["descriptor_path"]).parent
    assert (
        root / f"prediction_patches/{receipt.content_sha256}.json"
    ).read_bytes() == patch_bytes
    assert not paths.pipeline_reevaluation_patch_path.exists()
    runner, effective = _runner(paths, grown)
    assert isinstance(runner, AnemoneMorpionSearchRunner)
    runner.load_or_create(
        paths.work_dir / grown.latest_runtime_checkpoint_path, None, effective
    )
    restored = runner.prediction_origins()
    assert restored is not None and restored.current == second.current_origins
    applied_ids = next(
        event["written"]
        for event in read_object(root / "prediction_evidence.json")["events"].values()
        if event["kind"] == "accepted_patch"
    )
    nodes = native(runner)._nodes_by_public_id()
    assert all(
        nodes[key].tree_evaluation.direct_evaluation_version is None
        for key in applied_ids
    )
    runner.discard_uncommitted_runtime()
    # Existing full refresh remains a distinct explicit activation operation.
    request = prepare_model_activation_request(paths, training_generation=1)
    after = activate_model(paths, request)
    refreshed = tracked(paths)
    assert {origin.kind for origin in refreshed.current_origins.values()} == {
        "full_refresh"
    }
    assert {
        origin.evaluator_sha256 for origin in refreshed.current_origins.values()
    } == {request["candidate"]["sha256"]}
    assert len(refreshed.creation_generations) == len(second.creation_generations)
    assert after.generation == grown.generation + 1


def test_historical_checkpoint_remains_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old sources remain readable but never acquire fabricated origins from the model pin."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, state = setup_activation(tmp_path / "work")
    head = publication_records(state.metadata)[-1]
    assert read_committed_prediction_origins(paths.work_dir, head) is None
    after = grow(args, paths)
    assert "prediction_lineage" not in publication_records(after.metadata)[-1]


@pytest.mark.parametrize("after_commit", [False, True])
def test_commit_crash_never_authenticates_a_partial_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_commit: bool,
) -> None:
    """Fault the existing sole commit; recover by the accepted head, never an orphan."""
    from chipiron.environments.morpion.bootstrap import (
        checkpoint_patch_receipts as receipts,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, before = setup_activation(
        tmp_path / "work", prediction_origin_tracking=True
    )
    before_bytes = paths.run_state_path.read_bytes()
    assert propose(args).patch_written
    original = receipts.save_bootstrap_run_state

    def interrupted(state: Any, path: Path, **kwargs: Any) -> None:
        if state.generation > before.generation:
            if after_commit:
                original(state, path, **kwargs)
            message = "injected commit crash"
            raise RuntimeError(message)
        original(state, path, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(receipts, "save_bootstrap_run_state", interrupted)
        with pytest.raises(RuntimeError, match="injected commit crash"):
            grow(args, paths)
    survived = load_bootstrap_run_state(paths.run_state_path)
    survived_map = tracked(paths)
    if after_commit:
        assert survived.generation == before.generation + 1
        assert receipt_ledger(survived)
        before_recovery = paths.run_state_path.read_bytes()
        assert grow(args, paths, max_cycles=0) == survived
        assert paths.run_state_path.read_bytes() == before_recovery
        assert tracked(paths) == survived_map
    else:
        assert paths.run_state_path.read_bytes() == before_bytes
        assert {origin.kind for origin in survived_map.current_origins.values()} == {
            "genesis"
        }
        grow(args, paths)
        assert len(receipt_ledger(load_bootstrap_run_state(paths.run_state_path))) == 1
    assert not paths.pipeline_reevaluation_patch_path.exists()


@pytest.mark.parametrize("damage", ["missing", "changed", "uncommitted"])
def test_committed_reader_refuses_broken_or_uncommitted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    """An intact hash or an orphan directory alone is not source authentication."""
    local_leaderboard(monkeypatch, tmp_path)
    _, paths, state = setup_activation(
        tmp_path / "work", prediction_origin_tracking=True
    )
    head = publication_records(state.metadata)[-1]
    root = (paths.work_dir / head["descriptor_path"]).parent
    if damage == "missing":
        (root / "prediction_lineage.json").unlink()
    elif damage == "changed":
        with (root / "prediction_evidence.json").open("ab") as stream:
            stream.write(b" ")
    else:
        head = {**head, "descriptor_sha256": "a" * 64}
    with pytest.raises(ValueError):
        read_committed_prediction_origins(paths.work_dir, head)


@pytest.mark.parametrize("equal_score", [False, True])
def test_unobserved_write_and_evaluator_rebind_refuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    equal_score: bool,
) -> None:
    """Neither a current model pin nor scalar replacement authenticates a write."""
    from anemone.node_evaluation.common.canonical_value import make_estimate_value

    from tests.environments.test_morpion_value_certainty import neural_evaluator

    local_leaderboard(monkeypatch, tmp_path)
    _, paths, state = setup_activation(
        tmp_path / "work", prediction_origin_tracking=True
    )
    runner, effective = _runner(paths, state)
    assert isinstance(runner, AnemoneMorpionSearchRunner)
    runner.load_or_create(
        paths.work_dir / state.latest_runtime_checkpoint_path, None, effective
    )
    root = native(runner).tree.root_node
    old_value = root.tree_evaluation.direct_value
    root.tree_evaluation.direct_value = make_estimate_value(
        score=old_value.score if equal_score else -999
    )
    assert root.tree_evaluation.direct_value is not old_value
    with pytest.raises(ValueError, match="outside observed Growth writes"):
        runner.save_checkpoint(tmp_path / "unobserved")
    runner.discard_uncommitted_runtime()
    runner.load_or_create(
        paths.work_dir / state.latest_runtime_checkpoint_path, None, effective
    )
    native(runner).set_evaluator(neural_evaluator(99.0))
    with pytest.raises(ValueError, match="outside the owned model binding"):
        runner.grow(1)
    with pytest.raises(ValueError):
        runner.prediction_origins()


@pytest.mark.parametrize("protect", [False, True])
def test_native_equal_writes_protection_missing_and_certified_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
    protect: bool,
) -> None:
    """A tiny legal endgame uses native 0.4.1; no fabricated scientific genesis."""
    from anemone.node_evaluation.common.canonical_value import make_forced_value

    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionReevaluationPatch,
        MorpionReevaluationPatchRow,
    )
    from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
        GROWTH_NUMERIC_POLICY,
    )
    from chipiron.environments.morpion.bootstrap.runtime import runner as runtime_module
    from tests.environments.test_morpion_anemone_runner import _make_model_bundle

    bundle = _make_model_bundle(tmp_path / "model")
    runner = runtime_module.AnemoneMorpionSearchRunner(
        runtime_module.AnemoneMorpionSearchRunnerArgs(prediction_origin_tracking=True)
    )
    with monkeypatch.context() as setup:
        setup.setattr(runtime_module, "initial_state", lambda: endgame[0])
        runner.load_or_create(
            None,
            bundle,
            MorpionBootstrapEffectiveRuntimeConfig(1000, reevaluation_blend_alpha=1.0),
        )
    runner.grow(1)
    nodes: tuple[Any, ...] = tuple(runner.iter_profile_nodes())
    root = native(runner).tree.root_node
    terminal = next(node for node in nodes if node.state.is_game_over())
    if protect:
        # Observe a native certificate write. It must never become neural evidence.
        native(runner).tree_manager.node_evaluator._store_direct_value(
            node=root,
            value=make_forced_value(score=root.tree_evaluation.tree_value.score),
        )
    old_value = root.tree_evaluation.direct_value
    old_version = root.tree_evaluation.direct_evaluation_version
    terminal_value = terminal.tree_evaluation.direct_value
    row = MorpionReevaluationPatchRow(
        str(root.id),
        old_value.score,
        is_exact=False,
        is_terminal=False,
        metadata={
            "source": "active_model_reevaluation",
            "numeric_policy": GROWTH_NUMERIC_POLICY,
        },
    )
    reuse = MorpionReevaluationPatchRow(
        str(terminal.id),
        terminal_value.score,
        is_exact=True,
        is_terminal=True,
        metadata={"source": "terminal_existing_value"},
    )
    patch = MorpionReevaluationPatch(
        "actual-native",
        "now",
        1,
        "seed",
        "model",
        (row, reuse, replace(row, node_id="missing")),
        tree_generation=1,
    )
    assert runner.apply_reevaluation_patch(patch) == (1 if protect else 2)
    disposition = runner.last_prediction_patch_disposition()
    assert disposition is not None
    assert disposition.applied == (() if protect else (str(root.id),))
    assert disposition.protected == ((str(root.id),) if protect else ())
    assert disposition.missing == ("missing",)
    assert disposition.certified_reuse == (str(terminal.id),)
    assert root.tree_evaluation.direct_value == old_value
    assert terminal.tree_evaluation.direct_value == terminal_value
    assert root.tree_evaluation.direct_evaluation_version == (
        old_version if protect else None
    )
    tracking = runner.prediction_origins()
    assert tracking is not None
    assert str(terminal.id) not in tracking.writes
    if protect:
        assert str(root.id) not in tracking.writes
    else:
        assert tracking.writes[str(root.id)][0] == "accepted_patch"


def test_real_mixed_birth_map_remains_private_until_future_activation_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3 can seal actual A/B writes; this does not commit a model-only activation."""
    from chipiron.environments.morpion.bootstrap.generation_publication import (
        DurableGenerationPublication,
    )
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        growth_receipt_ownership,
        private_generation_validation,
    )
    from chipiron.environments.morpion.bootstrap.prediction_lineage import (
        read_prediction_lineage,
    )
    from chipiron.environments.morpion.bootstrap.prediction_origin_publication import (
        seal_prediction_origins,
    )
    from chipiron.environments.morpion.bootstrap.training_publication import (
        accepted_selected_model,
    )

    local_leaderboard(monkeypatch, tmp_path)
    _, paths, before = setup_activation(
        tmp_path / "work", prediction_origin_tracking=True
    )
    before = load_bootstrap_run_state(paths.run_state_path)
    authority = paths.run_state_path.read_bytes()
    old = tracked(paths)
    _, candidate = accepted_selected_model(paths, before, 1)
    current, _ = committed_evaluator(paths.work_dir, before)
    assert current is not None
    producer = publication_records(before.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    runner, effective = _runner(paths, before)
    assert isinstance(runner, AnemoneMorpionSearchRunner)
    with growth_receipt_ownership(paths):
        publication = DurableGenerationPublication(
            paths, scientific_v2_producer=producer, closed_scientific_snapshot=True
        )
        publication.prepare(before)
        model = publication.begin_cycle(current)
        runner.load_or_create(
            paths.work_dir / before.latest_runtime_checkpoint_path, model, effective
        )
        candidate_pin = publication.freeze_model(candidate)
        # Existing private rebind primitive only; no runtime activation API is added.
        runner._set_runtime_evaluator_from_bundle(candidate_pin, reevaluate_tree=False)
        runner.grow(1)
        checkpoint = publication.checkpoint_path(2, "sharded")
        runner.save_checkpoint(checkpoint)
        publication.producer = {**producer, "evaluator": candidate["sha256"]}
        assert publication.staging is not None
        with private_generation_validation(publication.staging):
            publication.sharded_path = runner.export_closed_scientific_snapshot(
                publication.staging / "scientific_snapshot",
                generation=2,
                producer=publication.producer,
                previous_producer=producer,
                committed_source=paths.work_dir / before.latest_tree_snapshot_path,
            )
        tracking = runner.prediction_origins()
        assert tracking is not None
        ref = seal_prediction_origins(tracking, publication, checkpoint)
        lineage = read_prediction_lineage(
            (publication.staging / ref["path"]).read_bytes(),
            sidecar_sha256=ref["sha256"],
            capsule_file_sha256=ref["sha256"],
        )
        assert {origin.evaluator_sha256 for origin in lineage.origins} == {
            current["sha256"],
            candidate["sha256"],
        }
        assert {origin.kind for origin in lineage.origins} == {
            "genesis",
            "growth_birth",
        }
        assert all(
            tracking.current[node] == origin
            for node, origin in old.current_origins.items()
        )
        assert lineage.default is not None and lineage.births
        publication.close()
    assert paths.run_state_path.read_bytes() == authority
    assert tracked(paths) == old


def test_rehashed_map_cannot_assign_an_unrelated_applied_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even internally supplied matching hashes cannot turn a proposal into a write."""
    import copy
    import json

    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        file_digest,
    )
    from chipiron.environments.morpion.bootstrap.prediction_origin_publication import (
        read_prediction_evidence,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(
        tmp_path / "work", prediction_origin_tracking=True
    )
    assert propose(args).patch_written
    state = grow(args, paths)
    head = copy.deepcopy(publication_records(state.metadata)[-1])
    root = (paths.work_dir / head["descriptor_path"]).parent
    descriptor = read_object(paths.work_dir / head["descriptor_path"])
    lineage_path = root / "prediction_lineage.json"
    lineage = read_object(lineage_path)
    index = next(
        index
        for index, origin in enumerate(lineage["origins"])
        if origin["kind"] == "accepted_patch"
    )
    actual = tracked(paths)
    untouched = next(
        node
        for node, origin in actual.current_origins.items()
        if origin.kind == "genesis"
    )
    lineage["overrides"] = [
        pair for pair in lineage["overrides"] if pair[0] != untouched
    ] + [[untouched, index]]
    lineage_path.write_text(json.dumps(lineage))
    descriptor["prediction_lineage"]["sha256"] = file_digest(lineage_path)
    head["prediction_lineage"] = descriptor["prediction_lineage"]
    for entry in descriptor["files"]:
        if entry["path"] == "prediction_lineage.json":
            entry.update(
                sha256=file_digest(lineage_path), size=lineage_path.stat().st_size
            )
    # Bypass only the outer seal in this negative structural test. The production
    # public reader rejects these changed bytes even earlier at the capsule hash.
    with pytest.raises(
        ValueError, match="fabricated from a proposed or unrelated write"
    ):
        read_prediction_evidence(paths.work_dir, head, descriptor)


@pytest.mark.parametrize("layout", ["sharded", "json-zst"])
@pytest.mark.parametrize("maximum", [4, 20])
def test_node_budget_rollback_discards_rejected_prediction_births(
    tmp_path: Path, layout: str, maximum: int
) -> None:
    """Existing rollback restores both native values and volatile write evidence."""
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        AnemoneMorpionSearchRunnerArgs,
    )
    from tests.environments.test_morpion_anemone_runner import _make_model_bundle
    from tests.environments.test_morpion_unique_node_budget import payload

    model = _make_model_bundle(tmp_path / "model")
    reference = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(runtime_checkpoint_format=layout)
    )
    bounded = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            runtime_checkpoint_format=layout,
            max_tree_nodes=maximum,
            prediction_origin_tracking=True,
        )
    )
    try:
        reference.load_or_create(None, model)
        bounded.load_or_create(None, model)
        expected = payload(reference)
        for _ in range(32):
            reference.grow(1)
            if reference.current_tree_size() > maximum:
                break
            expected = payload(reference)
            if reference.current_tree_size() == maximum:
                break
        bounded.grow(32)
        assert payload(bounded) == expected
        assert bounded.last_node_budget_stop_reason is not None
        tracker = bounded.prediction_origins()
        assert tracker is not None
        assert set(tracker.expected_values) == set(
            native(bounded)._nodes_by_public_id()
        )
        assert set(tracker.writes) <= set(tracker.expected_values)
        bounded.save_checkpoint(
            tmp_path / ("bounded" if layout == "sharded" else "bounded.json.zst")
        )
        assert tracker.saved is not None
    finally:
        bounded.discard_uncommitted_runtime()
        reference.discard_uncommitted_runtime()
