"""Serialized real owner transitions, not a claim of C7.1 worker overlap."""

import time
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import continuous_lifecycle
from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import prepared_dataset as prepared
from chipiron.environments.morpion.bootstrap import prepared_dataset_storage as storage
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_sha256,
    load_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.control import (
    apply_control_to_args,
    load_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    committed_evaluator,
    file_digest,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.managed_sessions import begin_owned
from chipiron.environments.morpion.bootstrap.model_activation import (
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.ordinary_growth import OrdinaryGrowthUpdate
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from chipiron.environments.morpion.bootstrap.worker_result_io import publication_record
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import run_dataset, setup_fenced
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_continuous_runtime import candidate_family, fixture
from tests.environments.test_morpion_continuous_runtime import (
    continuous as continuous,
)
from tests.environments.test_morpion_learning_schedule import monitored as monitored

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def staged_growth(
    args: Any, paths: Any, monkeypatch: pytest.MonkeyPatch, *, save: bool = True
) -> Iterator[tuple[GrowthPatchReceiptSession, OrdinaryGrowthUpdate]]:
    """Capture a real private result under the owner, then stage serial acceptances.

    Only this fixture releases ownership before acceptance. Production retains its
    broad lock and separately detects unexpected writes. No worker thread, sleeps,
    edited run-state input, synthetic checkpoint or mocked acceptance is involved.
    """
    source = load_bootstrap_run_state(paths.run_state_path)
    producer = publication_records(source.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    runner = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            runtime_checkpoint_format=args.runtime_checkpoint_format,
            random_seed=4,
            search_args=default_search_args(rollout=args.search.rollout),
        )
    )
    session = GrowthPatchReceiptSession(
        paths,
        runner,
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )
    try:
        with growth_receipt_ownership(paths), monkeypatch.context() as patch:
            source = session.prepare(source, args)
            config = load_bootstrap_config(paths.bootstrap_config_path)
            control = load_bootstrap_control(paths.control_path)
            if not save:
                patch.setattr(stages, "should_save_progress", lambda **_: False)
            result = stages._run_one_pipeline_growth_cycle(
                args=apply_control_to_args(args, control),
                paths=paths,
                runner=runner,
                run_state=source,
                control=control,
                config_hash=bootstrap_config_sha256(config),
                bootstrap_config=config,
                receipt_session=session,
            )
            assert result.generation == source.generation + int(save)
            assert result.cycle_index == source.cycle_index + 1
        yield session, OrdinaryGrowthUpdate(source, result)
    finally:
        session.discard_runtime()
        assert session.publication is not None
        session.publication.close()


def accept_growth(
    session: GrowthPatchReceiptSession, update: OrdinaryGrowthUpdate
) -> Any:
    """Use the production acceptance and real durable writer under its owner."""
    with growth_receipt_ownership(session.paths):
        state = session.commit_growth(update)
    assert load_bootstrap_run_state(session.paths.run_state_path) == state
    return state


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("order", ["dataset-first", "growth-first"])
def test_generation_100_dataset_acceptance_in_both_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    """Both 101 and the exact accepted 100 survive, followed by actual native Ready."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        evaluators=candidate_family(),
        learning_schedule=Policy(),
        dataset_cadence=100,
        training_cadence=100,
        maximum=100000,
        strict_execution=True,
        managed=True,
    )
    begin_owned(
        paths,
        load_bootstrap_config(paths.bootstrap_config_path),
        "42",
        time.monotonic(),
        600,
    )
    # Generate authentic consecutive history. Do not relabel a tiny genesis as 100.
    for generation in range(2, 101):
        assert grow(args).generation == generation
    original = stages.accept_dataset
    observed: dict[str, Any] = {}

    def accept(*a: Any, **kw: Any) -> Any:
        (tmp_path / "reloaded-before.json").write_text(paths.run_state_path.read_text())
        with staged_growth(args, paths, monkeypatch) as (session, update):
            debt = publication.load_frame(paths, update.source).state.learning_debt
            if order == "growth-first":
                accept_growth(session, update)
            result = original(*a, **kw)
            accepted = load_bootstrap_run_state(paths.run_state_path)
            reference = publication_record(accepted, 100)["accepted_dataset"]
            operational = storage.entries(accepted)
            frame = publication.load_frame(paths, accepted)
            assert frame.state.dataset_job is not None and frame.state.ready is None
            if order == "dataset-first":
                accept_growth(session, update)
            reloaded = load_bootstrap_run_state(paths.run_state_path)
            assert reloaded.generation == 101
            assert publication_record(reloaded, 100)["accepted_dataset"] == reference
            assert storage.entries(reloaded) == operational
            current = publication.load_frame(paths, reloaded)
            assert current.state.dataset_job == frame.state.dataset_job
            assert current.state.learning_debt == debt
            assert (
                committed_evaluator(paths.work_dir, reloaded)[0]
                == (committed_evaluator(paths.work_dir, update.source)[0])
            )
            observed.update(
                reference=reference, checkpoint=reloaded.latest_runtime_checkpoint_path
            )
            return result

    monkeypatch.setattr(stages, "accept_dataset", accept)
    run_dataset(args, generation=100)
    final = load_bootstrap_run_state(paths.run_state_path)
    assert final.generation == 101
    assert final.latest_runtime_checkpoint_path == observed["checkpoint"]
    assert publication_record(final, 100)["accepted_dataset"] == observed["reference"]
    frame = publication.load_frame(paths, final)
    assert frame.state.dataset_job is None and frame.state.ready is not None
    [operational] = storage.entries(final).values()
    storage.ready_descriptor(paths, operational)
    (tmp_path / "reloaded-after.json").write_text(paths.run_state_path.read_text())


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("order", ["ready-first", "growth-first"])
@pytest.mark.parametrize("save", [True, False])
def test_ready_and_growth_in_both_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str, *, save: bool
) -> None:
    """Preserve actual prepared bytes, reservations and debt in saved/unsaved cycles."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    original = prepared.prepare_accepted
    calls = []

    def complete(*a: Any, **kw: Any) -> None:
        with staged_growth(args, paths, monkeypatch, save=save) as (session, update):
            debt = publication.load_frame(paths, update.source).state.learning_debt
            if order == "growth-first":
                accept_growth(session, update)
            original(*a, **kw)
            ready = load_bootstrap_run_state(paths.run_state_path)
            frame = publication.load_frame(paths, ready)
            assert frame.state.ready is not None and frame.state.dataset_job is None
            if order == "ready-first":
                accept_growth(session, update)
            final = load_bootstrap_run_state(paths.run_state_path)
            current = publication.load_frame(paths, final)
            assert current.state.ready == frame.state.ready
            assert current.state.learning_debt == debt
            assert storage.entries(final) == storage.entries(ready)
            assert final.generation == generation + int(save)
            assert final.cycle_index == update.source.cycle_index + 1
            if not save:
                assert (
                    final.latest_runtime_checkpoint_path
                    == update.source.latest_runtime_checkpoint_path
                )
            [record] = storage.entries(final).values()
            storage.ready_descriptor(paths, record)
            calls.append(final.generation)

    monkeypatch.setattr(prepared, "prepare_accepted", complete)
    run_dataset(args, generation=generation)
    assert calls == [generation + int(save)]


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("order", ["training-first", "growth-first"])
def test_training_keeps_every_candidate_and_m1_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    """Whole-cohort acceptance survives Growth without replacing the computation model."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    with staged_growth(args, paths, monkeypatch) as (session, update):
        m1 = committed_evaluator(paths.work_dir, update.source)[0]
        if order == "growth-first":
            accept_growth(session, update)
        result = stages.run_pipeline_training_stage(args, generation=generation)
        assert result.training_status == "done"
        accepted = load_bootstrap_run_state(paths.run_state_path)
        reference = publication_record(accepted, generation)["accepted_training"]
        proposal = training._proposal(paths, reference)
        names = list(proposal["models"])
        assert set(names) == {"linear", "transformer_v1"}
        identities = {
            name: (
                artifact_digest(paths.work_dir / proposal["models"][name]["path"]),
                file_digest(
                    paths.work_dir / proposal["models"][name]["path"] / "optimizer.pt"
                ),
            )
            for name in names
        }
        parents = training.latest_candidate_parents(paths, accepted, names)
        frame = publication.load_frame(paths, accepted)
        assert frame.state.last_accepted_round == 1 and frame.state.cohort is None
        if order == "training-first":
            accept_growth(session, update)
        final = load_bootstrap_run_state(paths.run_state_path)
        current = publication.load_frame(paths, final)
        assert current.state.last_accepted_round == 1 and current.state.cohort is None
        assert current.state.pending_promotion == frame.state.pending_promotion
        assert current.state.promotion_debt == frame.state.promotion_debt
        assert committed_evaluator(paths.work_dir, final)[0] == m1
        assert publication_record(final, generation)["accepted_training"] == reference
        assert training.latest_candidate_parents(paths, final, names) == parents
        for name in names:
            bundle = paths.work_dir / proposal["models"][name]["path"]
            assert identities[name] == (
                artifact_digest(bundle),
                file_digest(bundle / "optimizer.pt"),
            )
            assert parents[name]["accepted_training"] == reference
        assert storage.entries(final) == storage.entries(accepted)
        # Growth may introduce new debt; it may never reset an outstanding anchor.
        if frame.state.learning_debt is not None:
            assert current.state.learning_debt == frame.state.learning_debt
        assert final.generation == generation + 1


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("save", [True, False])
def test_current_admission_gate_refuses_without_losing_ready_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, save: bool
) -> None:
    """A real close transition blocks either write set; owned resume permits a retry."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    with staged_growth(args, paths, monkeypatch, save=save) as (session, update):
        assert continuous_lifecycle.close_admission(paths)
        before = paths.run_state_path.read_bytes()
        with pytest.raises(ValueError, match="current learning or admission gate"):
            accept_growth(session, update)
        assert paths.run_state_path.read_bytes() == before
        with growth_receipt_ownership(paths):
            resumed = continuous_lifecycle.resume_owned(
                paths, load_bootstrap_run_state(paths.run_state_path)
            )
            save_bootstrap_run_state(resumed, paths.run_state_path, durable=True)
        accepted = accept_growth(session, update)
        assert storage.entries(accepted) == storage.entries(resumed)
        assert publication.load_frame(paths, accepted).state.ready == (
            publication.load_frame(paths, resumed).state.ready
        )


@pytest.mark.usefixtures("monitored")
@pytest.mark.parametrize("model_only", [False, True])
def test_changed_checkpoint_evaluator_refuses_and_fresh_growth_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, model_only: bool
) -> None:
    """Actual activation wins; stale M1 Growth cannot overwrite M2 or accepted Training."""
    args, paths, _ = setup_activation(
        tmp_path / "run", model_only_activation=model_only
    )
    with staged_growth(args, paths, monkeypatch) as (session, update):
        request = prepare_model_activation_request(paths, training_generation=1)
        activated = activate_model(paths, request)
        before = paths.run_state_path.read_bytes()
        with pytest.raises(ValueError, match="checkpoint, evaluator or cycle changed"):
            accept_growth(session, update)
        assert paths.run_state_path.read_bytes() == before
        assert load_bootstrap_run_state(paths.run_state_path) == activated
    with staged_growth(args, paths, monkeypatch) as (session, update):
        retried = accept_growth(session, update)
    assert retried.generation == activated.generation + 1
    assert committed_evaluator(paths.work_dir, retried)[0] == request["candidate"]
    assert (
        publication_record(retried, 1)["accepted_training"]
        == (publication_record(activated, 1)["accepted_training"])
    )


@pytest.mark.usefixtures("continuous")
def test_broad_production_owner_rejects_unexpected_external_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately unauthorized write tests refusal, never compatible publication."""
    args, paths = fixture(tmp_path / "run")
    original = stages._run_one_pipeline_growth_cycle
    unauthorized = []

    def interrupted(**kw: Any) -> Any:
        result = original(**kw)
        state = load_bootstrap_run_state(paths.run_state_path)
        save_bootstrap_run_state(
            replace(state, metadata={**state.metadata, "unauthorized_writer": True}),
            paths.run_state_path,
            durable=True,
        )
        unauthorized.append(paths.run_state_path.read_bytes())
        return result

    monkeypatch.setattr(stages, "_run_one_pipeline_growth_cycle", interrupted)
    with pytest.raises(ValueError, match="outside the Growth ownership lock"):
        grow(args)
    assert paths.run_state_path.read_bytes() == unauthorized[0]


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("save", [True, False])
def test_no_intervening_result_and_cold_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, save: bool
) -> None:
    """The same acceptance operation handles an unchanged authority and cold restart."""
    args, paths = fixture(tmp_path / "run")
    with staged_growth(args, paths, monkeypatch, save=save) as (session, update):
        assert update.apply(paths, update.source) == update.result
        state = accept_growth(session, update)
        previous = publication_records(update.source.metadata)
        assert publication_records(state.metadata)[: len(previous)] == previous
        assert state.generation == update.source.generation + int(save)
    restarted = grow(args)
    assert restarted.generation == state.generation + 1
    assert (
        committed_evaluator(paths.work_dir, restarted)[0]
        == (committed_evaluator(paths.work_dir, state)[0])
    )


@pytest.mark.parametrize(
    "field",
    [
        "bootstrap_config_hash",
        "atomic_reevaluation",
        "checkpoint_bound_reevaluation_v1",
        "unknown_metadata",
    ],
)
def test_changed_prerequisite_is_not_a_worker_update(
    tmp_path: Path, field: str
) -> None:
    """Malformed inputs fail before any durable write; no fake worker acceptance."""
    _args, paths, source = setup_fenced(tmp_path / "run")
    before = paths.run_state_path.read_bytes()
    changed = replace(source, metadata={**source.metadata, field: {"changed": True}})
    with pytest.raises(ValueError, match="prerequisite or unowned metadata"):
        OrdinaryGrowthUpdate(source, source).apply(paths, changed)
    assert paths.run_state_path.read_bytes() == before


def test_result_cannot_write_worker_authority(tmp_path: Path) -> None:
    """A captured result is never a whole-state replacement, even with no new workers."""
    _args, paths, source = setup_fenced(tmp_path / "run")
    before = paths.run_state_path.read_bytes()
    changed = replace(source, latest_model_bundle_paths={"invented": "candidate"})
    with pytest.raises(ValueError, match="unowned state field"):
        OrdinaryGrowthUpdate(source, changed).apply(paths, source)
    assert paths.run_state_path.read_bytes() == before
