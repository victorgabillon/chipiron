"""Finite real continuous failure boundaries, with immutable generated scientific inputs."""

import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import dataset_publication, promotion
from chipiron.environments.morpion.bootstrap import model_activation as activation
from chipiron.environments.morpion.bootstrap import prepared_dataset as prepared
from chipiron.environments.morpion.bootstrap import prepared_dataset_storage as storage
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.continuous_contract import (
    Comparison,
    ControlRecord,
    JobRecord,
    Reservation,
    record_ref,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_continuous_runtime import fixture
from tests.environments.test_morpion_learning_schedule import monitored as monitored
from tests.environments.test_morpion_prepared_dataset import child
from tests.environments.test_morpion_value_certainty import endgame as endgame

if TYPE_CHECKING:
    from pathlib import Path


def learn(args: Any, paths: Any) -> int:
    """Run one real committed-source, accepted-preparation and whole-cohort path."""
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    assert (
        stages.run_pipeline_training_stage(args, generation=generation).training_status
        == "done"
    )
    return generation


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("point", ["before-admission", "during-extraction"])
def test_offer_claim_publication_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    """A claim or orphan proposal cannot replace admitted authority; later offers only pend."""
    args, paths = fixture(tmp_path / "run")
    sources = [grow(args) for _ in range(3)]
    before = load_bootstrap_run_state(paths.run_state_path)
    frame = publication.load_frame(paths, before)
    assert frame.state.pending_source == publication.source(
        paths, before, sources[-1].generation
    )
    assert (
        frame.state.learning_debt.anchor_cycle
        == publication.source(paths, before, sources[0].generation).cycle
    )
    if point == "before-admission":

        def interrupt(*a: Any, **k: Any) -> None:
            raise RuntimeError("after_claim_before_save")

        monkeypatch.setattr(dataset_publication, "save_bootstrap_run_state", interrupt)
        with pytest.raises(RuntimeError, match="after_claim_before_save"):
            run_dataset(args, generation=before.generation)
        after = publication.load_frame(
            paths, load_bootstrap_run_state(paths.run_state_path)
        )
        assert after == frame
        assert paths.pipeline_dataset_claim_path_for_generation(
            before.generation
        ).exists()
    else:

        def during(*a: Any, **k: Any) -> None:
            admitted = load_bootstrap_run_state(paths.run_state_path)
            original = publication.load_frame(paths, admitted)
            grown = grow(args)
            current = publication.load_frame(paths, grown)
            assert current.state.dataset_job == original.state.dataset_job
            assert current.state.pending_source.generation == grown.generation
            assert current.state.learning_debt == frame.state.learning_debt
            raise RuntimeError("admitted_extraction_interrupted")

        monkeypatch.setattr(stages, "_run_pipeline_dataset_stage", during)
        with pytest.raises(RuntimeError, match="admitted_extraction_interrupted"):
            run_dataset(args, generation=before.generation)
        state = load_bootstrap_run_state(paths.run_state_path)
        frame = publication.load_frame(paths, state)
        job = publication.read_record(paths, frame.state.dataset_job, frame.context)
        assert isinstance(job, JobRecord)
        with pytest.raises((OSError, ValueError)):
            publication.authenticate_job(paths, state, job)
        assert not any(
            "accepted_dataset" in r for r in publication_records(state.metadata)
        )


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("point", ["copy", "native", "ready"])
def test_real_continuous_producer_death(tmp_path: Path, point: str) -> None:
    """Reap a real producer and retry the same accepted rows/path under the original clock."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    failed = child(args, generation, "produce", point=point)
    assert failed.returncode == 73, failed.stdout
    state = load_bootstrap_run_state(paths.run_state_path)
    [record] = storage.entries(state).values()
    original = record["accepted"], record["deadline"], record["utc_deadline"]
    restarted = child(args, generation, "produce")
    assert restarted.returncode == 0, restarted.stdout
    state = load_bootstrap_run_state(paths.run_state_path)
    [after] = storage.entries(state).values()
    assert (after["accepted"], after["deadline"], after["utc_deadline"]) == original
    frame = publication.load_frame(paths, state)
    assert frame.state.dataset_job is None and frame.state.ready is not None
    storage.ready_descriptor(paths, after)
    assert not torch.cuda.is_initialized()


@pytest.mark.usefixtures("continuous")
def test_same_dataset_preconsume_repair_and_postconsume_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair only before first reading; later corruption never switches or rebuilds inputs."""
    from chipiron.environments.morpion.bootstrap import continuous_inputs

    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    state = load_bootstrap_run_state(paths.run_state_path)
    [key] = storage.entries(state)
    root = storage.directory(paths, key)
    tensor = root / ".rows.jsonl.shared-preparation-v1/entry/flat.pt"
    original_bytes = tensor.read_bytes()
    original = continuous_inputs.recover_before_consumption
    repaired = False

    def damage_once(*a: Any, **k: Any) -> None:
        nonlocal repaired
        if not repaired:
            tensor.unlink()
            repaired = True
        original(*a, **k)

    with monkeypatch.context() as patch:
        patch.setattr(continuous_inputs, "recover_before_consumption", damage_once)
        stages.run_pipeline_training_stage(args, generation=generation)
    assert repaired and tensor.read_bytes() == original_bytes
    assert not prepared.retire(paths, key)  # Pending comparison still pins this root.
    tensor.write_bytes(b"corrupt-after-consumption")
    with pytest.raises(ValueError, match="after consumption"):
        promotion.service(paths, args, ttl_seconds=60, owner=None)
    assert (
        publication.load_frame(
            paths, load_bootstrap_run_state(paths.run_state_path)
        ).state.last_accepted_round
        == 1
    )


@pytest.mark.usefixtures("continuous")
def test_complete_cohort_acceptance_survives_projection_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact accepted retry performs no second update, even after projection failure."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)

    def interrupted(*a: Any, **k: Any) -> None:
        raise RuntimeError("after_accept_before_projection")

    with monkeypatch.context() as patch:
        patch.setattr(training, "_repair", interrupted)
        with pytest.raises(RuntimeError, match="after_accept_before_projection"):
            stages.run_pipeline_training_stage(args, generation=generation)
    state = load_bootstrap_run_state(paths.run_state_path)
    frame = publication.load_frame(paths, state)
    assert frame.state.last_accepted_round == 1 and frame.state.cohort is None
    authority = paths.run_state_path.read_bytes()
    from chipiron.environments.morpion.bootstrap import cycle_training

    with monkeypatch.context() as patch:
        patch.setattr(
            cycle_training,
            "train_and_select_evaluators_streaming",
            lambda **kw: pytest.fail("accepted retry trained"),
        )
        result = stages.run_pipeline_training_stage(args, generation=generation)
        assert result.training_status == "done"
    assert paths.run_state_path.read_bytes() == authority
    reference = next(
        r["accepted_training"]
        for r in publication_records(state.metadata)
        if r["generation"] == generation
    )
    proposal = copy.deepcopy(training._proposal(paths, reference))
    proposal["models"].pop(next(iter(proposal["models"])))
    with pytest.raises(ValueError):
        training._validate(paths, state, proposal, verify=True)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.usefixtures("continuous")
def test_oldest_reserved_comparison_and_two_root_backpressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later cohort cannot replace T1, and two pinned populations forbid a third root."""
    args, paths = fixture(tmp_path / "run")
    first = learn(args, paths)
    assert promotion.service(paths, args, ttl_seconds=60, owner=None) == first
    state = load_bootstrap_run_state(paths.run_state_path)
    reserved = promotion.reservation(paths, state)
    assert reserved is not None and isinstance(reserved.payload, Reservation)
    assert reserved.payload.phase == "VERIFIED"
    comparison = publication.read_record(
        paths, reserved.payload.comparison, reserved.context
    )
    assert isinstance(comparison, ControlRecord) and isinstance(
        comparison.payload, Comparison
    )
    assert comparison.payload.population == reserved.payload.population
    assert comparison.payload.incumbent == reserved.payload.incumbent
    with monkeypatch.context() as hold:
        hold.setattr(promotion, "activate_at_boundary", lambda paths: None)
        second = learn(args, paths)
        assert second > first
        state = load_bootstrap_run_state(paths.run_state_path)
        frame = publication.load_frame(paths, state)
        assert frame.state.reserved_promotion == record_ref(reserved)
        assert frame.state.pending_promotion is not None
        assert len(storage.entries(state)) == 2
        for key in storage.entries(state):
            assert not prepared.retire(paths, key)
        third = grow(args).generation
        with pytest.raises(ValueError, match="resource_backpressure"):
            run_dataset(args, generation=third)
        assert len(storage.entries(load_bootstrap_run_state(paths.run_state_path))) == 2
    assert (
        promotion.reservation(paths, load_bootstrap_run_state(paths.run_state_path))
        == reserved
    )


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("patch", [False, True])
@pytest.mark.parametrize(
    "point", ["before-drain", "after-drain", "before-commit", "after-commit"]
)
def test_reserved_cold_cutover_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
    patch: bool,
    point: str,
) -> None:
    """Replay real cold cutover without a second epoch or lost backed-up proof."""
    from valanga.evaluations import Certainty

    from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
        GrowthPatchReceiptSession,
    )
    from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
    from tests.environments.test_morpion_model_activation_edges import propose

    position, _ = endgame
    with monkeypatch.context() as initial:
        initial.setattr(runner_module, "initial_state", lambda: position)
        args, paths = fixture(tmp_path / "run", proof=True)
    generation = learn(args, paths)
    promotion.service(paths, args, ttl_seconds=60, owner=None)
    before = load_bootstrap_run_state(paths.run_state_path)
    reserved = promotion.reservation(paths, before)
    assert reserved is not None and isinstance(reserved.payload, Reservation)
    compared = publication.read_record(
        paths, reserved.payload.comparison, reserved.context
    )
    assert isinstance(compared, ControlRecord) and isinstance(
        compared.payload, Comparison
    )
    assert compared.payload.decision == "promote"
    if patch:
        assert propose(args).patch_written
    original_save = activation._save

    def save(*a: Any, **k: Any) -> Any:
        if point == "before-commit" and a[4] == "activation.save":
            raise RuntimeError("cutover_fault")
        result = original_save(*a, **k)
        if point == "after-commit" and a[4] == "activation.save":
            raise RuntimeError("cutover_fault")
        return result

    def fail(*a: Any, **k: Any) -> None:
        raise RuntimeError("cutover_fault")

    with monkeypatch.context() as fault:
        if point == "before-drain":
            fault.setattr(GrowthPatchReceiptSession, "apply", fail)
        elif point == "after-drain":
            fault.setattr(
                runner_module.AnemoneMorpionSearchRunner,
                "refresh_model_estimates",
                fail,
            )
        else:
            fault.setattr(activation, "_save", save)
        with pytest.raises(RuntimeError, match="cutover_fault"):
            promotion.activate_at_boundary(paths)
    promotion.activate_at_boundary(paths)
    after = load_bootstrap_run_state(paths.run_state_path)
    assert publication.load_frame(paths, after).state.incumbent_epoch == 1
    head = next(
        r
        for r in reversed(publication_records(after.metadata))
        if "activation_request_sha256" in r
    )
    completion = activation.activation_completion(paths, after, head)
    assert completion["request"]["training_generation"] == generation
    assert (
        completion["request"]["candidate"]
        == committed_evaluator(paths.work_dir, after)[0]
    )
    restored, effective = activation._runner(paths, after)
    try:
        restored.load_or_create(
            paths.work_dir / after.latest_runtime_checkpoint_path,
            committed_evaluator(paths.work_dir, after)[1],
            effective,
        )
        root = restored._require_runtime().tree.root_node
        assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
        assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
        assert root.tree_evaluation.effective_value.score == 35
        restored.validate_model_refresh_certificates()
    finally:
        restored.discard_uncommitted_runtime()
    authority = paths.run_state_path.read_bytes()
    activation.activate_model(paths, completion["request"])
    assert paths.run_state_path.read_bytes() == authority
    grown = grow(args)
    assert grown.cycle_index > after.cycle_index
