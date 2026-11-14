"""Accept complete independent candidate cohorts without activating an evaluator."""

import math
from typing import TYPE_CHECKING, Any

from . import continuous_publication as publication
from . import continuous_transitions as transitions
from . import prepared_dataset_storage as storage
from .continuous_contract import (
    Accepted,
    Budget,
    ControlRecord,
    Debt,
    Job,
    JobRecord,
    Opportunity,
    Population,
    ReadyRecord,
    Ref,
    record_ref,
)
from .continuous_dataset import boundary
from .generation_publication_io import object_digest, read_object, require

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import (
        MorpionPipelineStageClaim,
        MorpionPipelineTrainingStatusArtifact,
    )
    from .run_state import MorpionBootstrapRunState


def validate_configuration(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    config: dict[str, Any],
) -> None:
    """Keep continual policy and schedule identity fences explicit for context4."""
    from .config import load_bootstrap_config

    publication.load_frame(paths, state)
    saved = load_bootstrap_config(paths.bootstrap_config_path).metadata
    require(
        all(
            config.get(key) == saved.get(key)
            for key in ("continual_learning", "learning_schedule")
        ),
        "Continuous learner configuration changed.",
    )


def admitted(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
) -> JobRecord:
    """Resolve original parents and inputs without checking a later incumbent epoch."""
    frame = publication.load_frame(paths, state)
    job = publication.read_record(
        paths, Ref.from_metadata(admission["continuous_admission"]), frame.context
    )
    require(isinstance(job, JobRecord), "Missing continuous Training admission.")
    assert isinstance(job, JobRecord) and job.payload.parents is not None
    require(
        job.payload.stage == "TRAIN"
        and publication.read_ref(paths, job.payload.claim)["claim_id"]
        == admission["claim_id"]
        and job.payload.source.generation == admission["generation"]
        and job.payload.round_number == admission["round_number"]
        and job.payload.accepted_dataset
        == Accepted.from_metadata(admission["dataset"]["reference"])
        and publication.read_ref(paths, job.payload.parents) == admission["parents"],
        "Continuous cohort admission changed.",
    )
    publication.authenticate_source(paths, state, job.payload.source)
    return job


def population(
    paths: MorpionBootstrapPaths, ready: ReadyRecord, config: dict[str, Any]
) -> tuple[ControlRecord, dict[str, Any]]:
    """Bind the exact ordered rows, native split and explicit evaluation recipe."""
    split = read_object(
        storage.verify_reference(paths, ready.payload.split.to_metadata())
    )
    indices = split["indices"]
    require(
        isinstance(indices, list) and len(indices) > 0,
        "Empty continuous validation population.",
    )
    recipes = [candidate["args"] for candidate in config["candidates"]]
    first = recipes[0]
    require(
        all(
            (
                r["validation_fraction"],
                r["validation_seed"],
                r["stable_validation_split"],
            )
            == (first["validation_fraction"], first["validation_seed"], True)
            for r in recipes
        ),
        "Candidate split recipes differ.",
    )
    recipe = publication.put_proof(
        paths,
        {
            "kind": "continuous_common_validation_v1",
            "validation_fraction": first["validation_fraction"],
            "validation_seed": first["validation_seed"],
            "max_rows": config["max_rows"],
            "row_chunk_size": config["chunk_size"],
            "device": first["device"],
            "flat_batch_size": 64,
            "relational_batch_size": 8,
            "reproducibility": {
                key: value
                for key, value in config.items()
                if key.startswith("torch_") or key == "execution_profile"
            },
        },
    )
    record = ControlRecord(
        ready.context,
        Population(
            ready.payload.accepted_dataset,
            ready.payload.rows.sha256,
            ready.payload.split.sha256,
            len(indices),
            recipe,
        ),
    )
    metadata = {
        "policy": "canonical_state_sha256_v1",
        "seed": first["validation_seed"],
        "fraction": first["validation_fraction"],
        "row_count": split["rows"],
        "train_count": split["rows"] - len(indices),
        "validation_count": len(indices),
        "validation_population_sha256": split["population"],
    }
    return record, metadata


def reader_job(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
    claim: MorpionPipelineStageClaim,
    ready: ReadyRecord,
    *,
    reserved_id: str | None = None,
) -> JobRecord:
    """Capture this serial learner's original finite clock, claim and own parents."""
    from .learning_policy import policy_from_metadata
    from .managed_sessions import KEY as MANAGED
    from .persistent_worker_state import monitored_identity
    from .prepared_dataset_intent import original_deadline

    frame = publication.load_frame(paths, state)
    policy = policy_from_metadata(admission["configuration"])
    require(policy is not None, "Missing learner resource limits.")
    assert policy is not None
    from .continuous_resources import reserve

    output_bytes, scratch_bytes = reserve(
        paths, state, policy, dataset=False, comparison=reserved_id is not None
    )
    owner = monitored_identity()
    end, wall = original_deadline(paths, policy.worker_max_seconds, claim, owner)
    launch = publication.put_proof(paths, state.metadata[MANAGED]["active"])
    clock = publication.put_proof(
        paths, {"owner": owner, "deadline": end, "utc_deadline": wall}
    )
    budget = publication.put_record(
        paths,
        ControlRecord(
            frame.context,
            Budget(
                launch,
                policy.worker_max_seconds,
                0.0,
                clock,
                output_bytes,
                scratch_bytes,
                object_digest(policy.to_metadata()),
            ),
        ),
    )
    return JobRecord(
        frame.context,
        Job(
            admission["attempt_id"],
            "TRAIN" if reserved_id is None else "COMPARE",
            ready.payload.source,
            publication.put_proof(
                paths,
                read_object(
                    paths.pipeline_training_claim_path_for_generation(claim.generation)
                ),
            ),
            launch,
            budget,
            "ADMITTED",
            ready.payload.accepted_dataset,
            record_ref(ready),
            publication.put_proof(paths, admission["parents"]),
            admission["attempt_id"],
            admission["round_number"],
            reserved_id,
        ),
    )


def admit_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
    claim: MorpionPipelineStageClaim,
) -> tuple[MorpionBootstrapRunState, dict[str, Any]]:
    """Consume the newest complete eligible Ready exactly once with current parents."""
    frame = publication.load_frame(paths, state)
    if frame.state.cohort is not None:
        from .continuous_lifecycle import previous_admission, retry_owned

        previous = publication.read_record(paths, frame.state.cohort, frame.context)
        assert isinstance(previous, JobRecord)
        original = previous_admission(paths, state, previous)
        state, retried = retry_owned(paths, state, previous, claim)
        return state, {
            **original,
            "attempt_id": admission["attempt_id"],
            "claim_id": claim.claim_id,
            "continuous_admission": record_ref(retried).to_metadata(),
        }
    require(frame.state.ready is not None, "No authoritative continuous Ready.")
    assert frame.state.ready is not None
    ready = publication.read_record(paths, frame.state.ready, frame.context)
    require(isinstance(ready, ReadyRecord), "Invalid Ready slot.")
    assert isinstance(ready, ReadyRecord)
    require(
        ready.payload.source.generation == admission["generation"],
        "Training selected another Dataset.",
    )
    pop, _ = population(paths, ready, admission["configuration"])
    assert isinstance(pop.payload, Population)
    job = reader_job(paths, state, admission, claim, ready)
    updated = publication.apply_owned(
        paths,
        state,
        boundary(paths, state),
        transitions.AdmitCohort(
            job=job, ready=ready, validation_count=pop.payload.validation_count
        ),
    )
    require(
        not isinstance(updated, transitions.NoWork),
        "Continuous learner admission refused.",
    )
    assert not isinstance(updated, transitions.NoWork)
    publication.put_record(paths, pop)
    return updated, {
        **admission,
        "schema_version": 4,
        "continuous_admission": record_ref(job).to_metadata(),
    }


def learned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
    results: Any,
    private_paths: MorpionBootstrapPaths,
) -> dict[str, Any]:
    """Report finite own-candidate metrics without any incumbent-selection authority."""
    job = admitted(paths, state, admission)
    assert job.payload.ready is not None
    ready = publication.read_record(paths, job.payload.ready, job.context)
    assert isinstance(ready, ReadyRecord)
    pop, expected = population(paths, ready, admission["configuration"])
    losses: dict[str, float] = {}
    for name, relative in results.model_bundle_paths.items():
        bundle = private_paths.resolve_work_dir_path(relative)
        assert bundle is not None
        manifest = read_object(bundle / "morpion_manifest.json")
        require(
            manifest["metadata"]["state_split"] == expected,
            "Candidate common population differs.",
        )
        loss = results.evaluator_metrics[name].validation_loss
        require(
            loss is not None and math.isfinite(loss),
            "Nonfinite candidate validation loss.",
        )
        losses[name] = float(loss)
    winner = min(losses, key=lambda name: (losses[name], name))
    require(
        winner == results.selected_evaluator_name, "Nondeterministic cohort winner."
    )
    return {
        "round_number": job.payload.round_number,
        "candidate": winner,
        "candidate_validation_losses": losses,
        "population": publication.put_record(paths, pop).to_metadata(),
        "validation_population": expected,
    }


def validate_learned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
    status: MorpionPipelineTrainingStatusArtifact,
) -> None:
    """Validate accepted retry from compact immutable proof, without cache tensors."""
    job = admitted(paths, state, admission)
    report = status.metadata.get("learned_cohort_v1")
    require(
        isinstance(report, dict) and "continual_selection" not in status.metadata,
        "Continuous cohort must not activate a model.",
    )
    assert isinstance(report, dict)
    require(
        set(report)
        == {
            "round_number",
            "candidate",
            "candidate_validation_losses",
            "population",
            "validation_population",
        },
        "Incomplete learned cohort.",
    )
    losses = {
        name: result.validation_loss
        for name, result in status.evaluator_results.items()
    }
    require(
        all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in losses.values()
        ),
        "Nonfinite cohort loss.",
    )
    finite = {name: float(value) for name, value in losses.items() if value is not None}
    winner = min(finite, key=lambda name: (finite[name], name))
    pop = publication.read_record(
        paths, Ref.from_metadata(report["population"]), job.context
    )
    require(
        isinstance(pop, ControlRecord) and isinstance(pop.payload, Population),
        "Missing common population proof.",
    )
    assert isinstance(pop, ControlRecord) and isinstance(pop.payload, Population)
    require(
        report["round_number"] == job.payload.round_number
        and report["candidate"] == status.selected_evaluator_name == winner
        and report["candidate_validation_losses"] == losses
        and pop.payload.accepted_dataset == job.payload.accepted_dataset
        and pop.payload.rows_sha256 == admission["dataset"]["rows_sha256"]
        and pop.payload.validation_count
        == report["validation_population"]["validation_count"]
        and all(
            result.num_validation_samples == pop.payload.validation_count
            for result in status.evaluator_results.values()
        ),
        "Learned cohort evidence changed.",
    )


def accept_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    proposal: dict[str, Any],
    reference: dict[str, Any],
) -> MorpionBootstrapRunState:
    """Close the cohort and enqueue cadence-eligible promotion in the acceptance save."""
    admission = proposal["admission"]
    job = admitted(paths, state, admission)
    assert job.payload.ready is not None and job.payload.accepted_dataset is not None
    ready = publication.read_record(paths, job.payload.ready, job.context)
    assert isinstance(ready, ReadyRecord)
    from .continuous_inputs import has_consumed

    require(
        has_consumed(paths, state, ready, job),
        "Cohort has no authenticated consumed-input proof.",
    )
    operational = storage.entries(state).get(
        job.payload.accepted_dataset.descriptor_sha256
    )
    require(operational is not None, "Consumed inputs disappeared before acceptance.")
    assert operational is not None
    storage.ready_descriptor(paths, operational)
    pop, population_metadata = population(paths, ready, admission["configuration"])
    report = proposal["status"]["metadata"]["learned_cohort_v1"]
    require(
        report["population"] == record_ref(pop).to_metadata()
        and report["validation_population"] == population_metadata,
        "Consumed population changed.",
    )
    frame = publication.load_frame(paths, state)
    policy = admission["configuration"]["continual_learning"]
    number = admission["round_number"]
    eligible = (
        number % policy["selection_every_rounds"] == 0
        and number % policy["activation_every_rounds"] == 0
    )
    opportunity = None
    if eligible:
        name = report["candidate"]
        candidate = publication.put_proof(
            paths,
            {
                **proposal["models"][name],
                "generation": admission["generation"],
                "name": name,
                "source": "local_training",
                "source_generation": admission["generation"],
                "local_trained_generation": admission["generation"],
            },
        )
        debt = Debt(admission["attempt_id"], state.cycle_index)
        if frame.state.pending_promotion is not None:
            old = publication.read_record(
                paths, frame.state.pending_promotion, frame.context
            )
            assert isinstance(old, ControlRecord) and isinstance(
                old.payload, Opportunity
            )
            debt = old.payload.debt
        opportunity = ControlRecord(
            frame.context,
            Opportunity(
                admission["attempt_id"],
                Accepted.from_metadata(reference),
                admission["generation"],
                number,
                name,
                candidate,
                job.payload.accepted_dataset,
                debt,
            ),
        )
    updated = publication.apply_owned(
        paths,
        state,
        boundary(paths, state),
        transitions.AcceptCohort(
            job,
            Accepted.from_metadata(reference),
            opportunity,
            publication.next_learning_debt(paths, state),
        ),
    )
    require(not isinstance(updated, transitions.NoWork), "Cohort completion suspended.")
    assert not isinstance(updated, transitions.NoWork)
    return updated
