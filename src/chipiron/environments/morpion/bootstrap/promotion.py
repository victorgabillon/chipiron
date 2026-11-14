"""Service the oldest reservation on the existing serial learner and Growth owner."""

import math
import uuid
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

from . import continuous_publication as publication
from . import continuous_transitions as transitions
from .continuous_contract import (
    Comparison,
    ControlRecord,
    Disposition,
    JobRecord,
    Opportunity,
    Population,
    ReadyRecord,
    Ref,
    Reservation,
    record_ref,
)
from .continuous_dataset import boundary
from .generation_publication_io import (
    PUBLICATION_KEY,
    artifact_digest,
    committed_evaluator,
    growth_receipt_ownership,
    require,
)
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState


def reservation(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> ControlRecord | None:
    """Resolve the bounded reservation, never the latest accepted training round."""
    frame = publication.load_frame(paths, state)
    if frame.state.reserved_promotion is None:
        return None
    record = publication.read_record(
        paths, frame.state.reserved_promotion, frame.context
    )
    require(
        isinstance(record, ControlRecord) and isinstance(record.payload, Reservation),
        "Invalid promotion reservation.",
    )
    assert isinstance(record, ControlRecord)
    return record


def _apply(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    action: transitions.Transition,
) -> MorpionBootstrapRunState:
    updated = publication.apply_owned(paths, state, boundary(paths, state), action)
    require(
        not isinstance(updated, transitions.NoWork), "Promotion transition refused."
    )
    assert not isinstance(updated, transitions.NoWork)
    return updated


def reserve_owned(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> MorpionBootstrapRunState:
    """Pin the pending opportunity before any later cohort can replace it."""
    from .training_publication import _proposal

    frame = publication.load_frame(paths, state)
    if (
        frame.state.reserved_promotion is not None
        or frame.state.pending_promotion is None
    ):
        return state
    record = publication.read_record(
        paths, frame.state.pending_promotion, frame.context
    )
    assert isinstance(record, ControlRecord) and isinstance(record.payload, Opportunity)
    opportunity = record.payload
    proposal = _proposal(paths, opportunity.accepted_training.to_metadata())
    pop = Ref.from_metadata(
        proposal["status"]["metadata"]["learned_cohort_v1"]["population"]
    )
    model = committed_evaluator(paths.work_dir, state)[0]
    require(model is not None, "Continuous comparison requires an incumbent.")
    assert model is not None
    incumbent = publication.put_proof(paths, model)
    reserved = ControlRecord(
        frame.context,
        Reservation(
            frame.state.pending_promotion,
            incumbent,
            pop,
            None,
            None,
            None,
            None,
            "RESERVED",
        ),
    )
    return _apply(paths, state, transitions.ReservePromotion(reserved))


def _evaluate(
    paths: MorpionBootstrapPaths,
    args: MorpionBootstrapArgs,
    rows: Path,
    model_ref: Ref,
    expected_population: dict[str, Any],
) -> float:
    """Run both frozen models with the same explicit evaluation recipe and population."""
    import torch

    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        load_morpion_model_bundle,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.args import (
        MorpionTrainingArgs,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.evaluation import (
        evaluate_streaming_metrics,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.state_split import (
        prepare_state_split,
    )
    from chipiron.learning.torch_runtime import resolve_torch_device

    model = publication.read_ref(paths, model_ref)
    bundle = paths.resolve_work_dir_path(model["path"])
    require(
        bundle is not None and artifact_digest(bundle) == model["sha256"],
        "Comparison model bytes changed.",
    )
    assert bundle is not None
    with torch.random.fork_rng(devices=[]):
        network, model_args, manifest = load_morpion_model_bundle(bundle)
    require(
        manifest.target_kind == "backup_value", "Comparison target contract differs."
    )
    keywords = {
        f.name: getattr(model_args, f.name)
        for f in fields(MorpionTrainingArgs)
        if hasattr(model_args, f.name)
    }
    evaluation = MorpionTrainingArgs(
        **keywords,
        dataset_file=rows,
        output_dir=rows.parent / "unused-comparison",
        batch_size=64 if model_args.model_kind in {"linear", "mlp"} else 8,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        stable_validation_split=True,
        device=args.training_device,
    )
    evaluation, actual = prepare_state_split(evaluation, args.training_max_rows)
    require(actual == expected_population, "Comparison population differs.")
    device = resolve_torch_device(args.training_device)
    stats = evaluate_streaming_metrics(
        model=network.to(device),
        args=evaluation,
        row_chunk_size=args.training_row_chunk_size,
        max_rows=args.training_max_rows,
        split="validation",
        device=device,
    )
    require(
        stats.sample_count == actual["validation_count"] and math.isfinite(stats.loss),
        "Invalid common-population loss.",
    )
    return float(stats.loss)


def service(
    paths: MorpionBootstrapPaths,
    args: MorpionBootstrapArgs,
    *,
    ttl_seconds: float,
    owner: str | None,
) -> int | None:
    """Run one reserved comparison before admitting any new Training cohort."""
    from .continuous_training import (
        admitted,
        population,
        reader_job,
        validate_configuration,
    )
    from .learning_worker import claim_metadata
    from .pipeline_claims import (
        claim_pipeline_stage,
        release_owned_pipeline_stage_claim,
    )
    from .prepared_dataset import borrow_continuous
    from .training_publication import _proposal

    with growth_receipt_ownership(paths, wait=True):
        state = load_bootstrap_run_state(paths.run_state_path)
        frame = publication.load_frame(paths, state)
        if not frame.state.admission_open or frame.state.cohort is not None:
            return None
        previous = state
        state = reserve_owned(paths, state)
        if state != previous:
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
        record = reservation(paths, state)
        if record is None:
            return None
        assert isinstance(record.payload, Reservation)
        value = record.payload
        if value.phase in {"VERIFIED", "CUTOVER"}:
            return None
        require(value.phase in {"RESERVED", "COMPARING"}, "Invalid comparison phase.")
        opportunity_record = publication.read_record(
            paths, value.opportunity, record.context
        )
        assert isinstance(opportunity_record, ControlRecord) and isinstance(
            opportunity_record.payload, Opportunity
        )
        opportunity = opportunity_record.payload
        proposal = _proposal(paths, opportunity.accepted_training.to_metadata())
        admission = proposal["admission"]
        validate_configuration(paths, state, admission["configuration"])
        learned_job = admitted(paths, state, admission)
        assert learned_job.payload.ready is not None
        ready = publication.read_record(
            paths, learned_job.payload.ready, record.context
        )
        assert isinstance(ready, ReadyRecord)
        pop, expected = population(paths, ready, admission["configuration"])
        require(record_ref(pop) == value.population, "Reserved population differs.")
        assert isinstance(pop.payload, Population)
        recipe = publication.read_ref(paths, pop.payload.evaluation_recipe)
        require(
            (
                recipe["validation_fraction"],
                recipe["validation_seed"],
                recipe["max_rows"],
                recipe["row_chunk_size"],
                recipe["device"],
            )
            == (
                args.validation_fraction,
                args.validation_seed,
                args.training_max_rows,
                args.training_row_chunk_size,
                args.training_device,
            ),
            "Comparison execution recipe changed.",
        )
        generation = opportunity.generation
        claim_path = paths.pipeline_training_claim_path_for_generation(generation)
        from .continuous_lifecycle import release_dead_claim, remaining, retry_owned

        prior_job = None
        if value.phase == "COMPARING":
            assert value.comparison_job is not None
            prior_job = publication.read_record(
                paths, value.comparison_job, record.context
            )
            assert isinstance(prior_job, JobRecord)
            remaining(paths, prior_job)
            release_dead_claim(paths, prior_job)
        metadata = claim_metadata(paths, state, generation, claim_path)
        claim = claim_pipeline_stage(
            generation=generation,
            stage="training",
            claim_path=claim_path,
            ttl_seconds=ttl_seconds,
            owner=owner,
            metadata=metadata,
        )
        job = reader_job(
            paths,
            state,
            {**admission, "attempt_id": uuid.uuid4().hex},
            claim,
            ready,
            reserved_id=value.opportunity.sha256,
        )
        if prior_job is not None:
            assert isinstance(prior_job, JobRecord)
            state, job = retry_owned(paths, state, prior_job, claim)
        else:
            if value.previous is not None:
                predecessor = publication.read_record(
                    paths, value.previous, record.context
                )
                assert isinstance(predecessor, ControlRecord) and isinstance(
                    predecessor.payload, Reservation
                )
                assert predecessor.payload.comparison_job is not None
                original = publication.read_record(
                    paths, predecessor.payload.comparison_job, record.context
                )
                assert isinstance(original, JobRecord)
                old_budget = remaining(paths, original)
                from .continuous_contract import Budget

                fresh = publication.read_record(
                    paths, job.payload.budget, record.context
                )
                assert isinstance(fresh, ControlRecord) and isinstance(
                    fresh.payload, Budget
                )
                inherited = publication.put_record(
                    paths,
                    ControlRecord(
                        record.context, replace(old_budget, launch=fresh.payload.launch)
                    ),
                )
                job = JobRecord(job.context, replace(job.payload, budget=inherited))
            state = _apply(paths, state, transitions.StartComparison(record, job))
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    try:
        with borrow_continuous(paths, args, job, ready) as (rows, _prepared):
            candidate_loss = _evaluate(
                paths, args, rows, opportunity.candidate, expected
            )
            incumbent_loss = _evaluate(paths, args, rows, value.incumbent, expected)
        comparison = ControlRecord(
            record.context,
            Comparison(
                value.opportunity.sha256,
                job.payload.claim,
                opportunity.candidate,
                value.incumbent,
                value.population,
                candidate_loss,
                incumbent_loss,
                "promote" if candidate_loss < incumbent_loss else "retain",
                "incumbent_on_equal_else_candidate_name",
            ),
        )
        with growth_receipt_ownership(paths, wait=True):
            state = load_bootstrap_run_state(paths.run_state_path)
            publication.authenticate_job(paths, state, job)
            current = reservation(paths, state)
            assert current is not None
            state = _apply(
                paths, state, transitions.VerifyComparison(current, comparison)
            )
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
        return generation
    finally:
        with growth_receipt_ownership(paths, wait=True):
            release_owned_pipeline_stage_claim(
                claim_path=claim_path, claim_id=claim.claim_id
            )


def request_bindings(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, generation: int
) -> dict[str, Any]:
    """Allow activation only for this verified reservation and current incumbent."""
    record = reservation(paths, state)
    require(record is not None, "No reserved promotion.")
    assert record is not None and isinstance(record.payload, Reservation)
    value = record.payload
    require(
        value.phase in {"VERIFIED", "CUTOVER"} and value.comparison is not None,
        "Unverified promotion.",
    )
    assert value.comparison is not None
    comparison = publication.read_record(paths, value.comparison, record.context)
    assert isinstance(comparison, ControlRecord) and isinstance(
        comparison.payload, Comparison
    )
    opportunity = publication.read_record(paths, value.opportunity, record.context)
    assert isinstance(opportunity, ControlRecord) and isinstance(
        opportunity.payload, Opportunity
    )
    require(
        generation == opportunity.payload.generation
        and comparison.payload.decision == "promote"
        and publication.read_ref(paths, value.incumbent)
        == committed_evaluator(paths.work_dir, state)[0],
        "Reserved comparison no longer authorizes cutover.",
    )
    root = value.previous if value.phase == "CUTOVER" else record_ref(record)
    require(root is not None, "Missing verified cutover predecessor.")
    assert root is not None
    return {
        "schema_version": 2,
        "reserved_promotion": root.to_metadata(),
        "promotion_comparison": value.comparison.to_metadata(),
    }


def resolve_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    *,
    promoted: bool,
    request: dict[str, Any] | None = None,
) -> MorpionBootstrapRunState:
    """Join retention or the cold publication's epoch increment to one durable save."""
    record = reservation(paths, state)
    assert record is not None and isinstance(record.payload, Reservation)
    value = record.payload
    assert value.comparison is not None
    comparison = publication.read_record(paths, value.comparison, record.context)
    assert isinstance(comparison, ControlRecord)
    if promoted:
        require(
            value.phase == "CUTOVER" and value.activation_request is not None,
            "Cold cutover was not reserved.",
        )
        assert value.activation_request is not None
        require(
            publication.read_ref(paths, value.activation_request) == request,
            "Cold request differs from reservation.",
        )
    else:
        require(
            publication.read_ref(paths, value.incumbent)
            == committed_evaluator(paths.work_dir, state)[0],
            "Retention comparison needs revalidation.",
        )
    updated = _apply(
        paths, state, transitions.ResolvePromotion(record, comparison, promoted)
    )
    if not promoted:
        updated = publication.attach_disposition(
            paths, updated, Disposition(value.opportunity, "retained", None, None)
        )
    return updated


def activate_at_boundary(paths: MorpionBootstrapPaths) -> None:
    """Use the unchanged synchronous cold-refresh implementation for the reservation."""
    from .model_activation import (
        _renew_automatic_request_after_drain,
        _request,
        activate_model,
    )

    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        record = reservation(paths, state)
        if record is None:
            return
        assert isinstance(record.payload, Reservation)
        value = record.payload
        if value.phase not in {"VERIFIED", "CUTOVER"}:
            return
        assert value.comparison is not None
        comparison = publication.read_record(paths, value.comparison, record.context)
        assert isinstance(comparison, ControlRecord) and isinstance(
            comparison.payload, Comparison
        )
        model = committed_evaluator(paths.work_dir, state)[0]
        require(model is not None, "Missing continuous incumbent.")
        assert model is not None
        current_incumbent = publication.put_proof(paths, model)
        if value.phase == "VERIFIED" and current_incumbent != value.incumbent:
            state = _apply(
                paths, state, transitions.Recompare(record, current_incumbent)
            )
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
            return
        if comparison.payload.decision == "retain":
            state = resolve_owned(paths, state, promoted=False)
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
            return
        opportunity = publication.read_record(paths, value.opportunity, record.context)
        assert isinstance(opportunity, ControlRecord) and isinstance(
            opportunity.payload, Opportunity
        )
        if value.activation_request is None:
            request = _request(paths, state, opportunity.payload.generation)
        else:
            request = publication.read_ref(paths, value.activation_request)
            if request["expected_head"] != state.metadata[PUBLICATION_KEY]["head_id"]:
                request = _renew_automatic_request_after_drain(paths, state, request)
        state = _apply(
            paths,
            state,
            transitions.BeginCutover(record, publication.put_proof(paths, request)),
        )
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    activate_model(paths, request)
