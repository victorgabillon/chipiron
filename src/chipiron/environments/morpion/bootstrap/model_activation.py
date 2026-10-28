"""Explicit accepted-model activation under the existing Growth commit authority.

Only fresh format-3/default-max/Linoo workspaces opt in at genesis. The request is
an optimistic binding, not a journal. Pending M1 work commits first; activation
then owns one private runtime and publishes through the ordinary capsule/receipt
session. Failure discards RAM. Recovery reads the existing authoritative head.
"""

import copy
import logging
import time
from dataclasses import asdict, replace
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import torch

from chipiron.environments.morpion.learning.target_confidence import (
    validate_confidence_policy,
)

from .checkpoint_patch_receipts import (
    RECEIPTS_METADATA_KEY,
    GrowthPatchReceiptSession,
    current_model_identity,
    receipt_ledger,
)
from .config import load_bootstrap_config
from .control import (
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    MorpionBootstrapEffectiveRuntimeConfig,
)
from .derived.provenance import inside
from .generation_publication_io import (
    PUBLICATION_KEY,
    capsule_path,
    committed_evaluator,
    growth_receipt_ownership,
    object_digest,
    publication_records,
    read_committed_descriptor,
    read_object,
    require,
)
from .performance import StageMeasurement, persist_stage_measurement
from .reevaluation_transaction import (
    GROWTH_NUMERIC_POLICY,
    cursor_digest,
    transaction_state,
)
from .run_state import load_bootstrap_run_state
from .runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from .training_publication import accepted_selected_model
from .worker_result_io import workspace_epoch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState

POLICY = {
    "checkpoint_format": 3,
    "search": "morpion_default_max_linoo_alternating_all_children_v1",
    "confidence": "selected_value_v2",
    "numeric": GROWTH_NUMERIC_POLICY,
    "refresh_scope": "all",
    "batch_size": 64,
}
LOGGER = logging.getLogger(__name__)


def _runner(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> tuple[AnemoneMorpionSearchRunner, MorpionBootstrapEffectiveRuntimeConfig]:
    config = load_bootstrap_config(paths.bootstrap_config_path)
    validate_confidence_policy("selected_value_v2", config.dataset.family_target_policy)
    require(
        config.search.rollout.max_extra_steps is not None
        or not config.search.rollout.enabled,
        "Activation admits bounded rollout only.",
    )
    effective = state.metadata.get(BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY)
    runtime_config = (
        MorpionBootstrapEffectiveRuntimeConfig(config.runtime.tree_branch_limit)
        if effective is None
        else MorpionBootstrapEffectiveRuntimeConfig(**effective)
    )
    return AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            search_args=default_search_args(rollout=config.search.rollout),
            runtime_checkpoint_format=config.runtime.runtime_checkpoint_format,
        )
    ), runtime_config


def validate_activation_genesis(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    producer: Mapping[str, str],
) -> dict[str, Any]:
    """Cold-check the explicitly new workspace; no historical adoption or rewriting."""
    model = current_model_identity(paths)
    require(
        producer["evaluator"] == model["sha256"],
        "Genesis producer must bind the full model digest.",
    )
    runner, effective = _runner(paths, state)
    assert state.latest_runtime_checkpoint_path is not None
    try:
        with torch.random.fork_rng(devices=[]):
            runner.load_or_create(
                inside(paths.work_dir, state.latest_runtime_checkpoint_path),
                inside(paths.work_dir, model["path"]),
                effective,
            )
        runner.validate_model_refresh_certificates()
    finally:
        runner.discard_uncommitted_runtime()
    return model


def _request(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    training_generation: int,
) -> dict[str, Any]:
    require(
        isinstance(training_generation, int)
        and not isinstance(training_generation, bool)
        and training_generation >= 1,
        "Explicit integer Training generation required.",
    )
    require(
        state.metadata.get(PUBLICATION_KEY, {}).get("evaluator_contract") == 1,
        "Activation requires an explicitly initialized format-3 workspace.",
    )
    transaction = transaction_state(state)
    require(
        transaction is not None and transaction["schema_version"] == 2,
        "Activation requires Growth-compatible atomic numeric schema 2.",
    )
    assert transaction is not None
    receipt_ledger(state)
    model, _ = committed_evaluator(paths.work_dir, state)
    accepted, candidate = accepted_selected_model(paths, state, training_generation)
    require(model != candidate, "Selected model is already active.")
    return {
        "schema_version": 1,
        "identity": workspace_epoch(state),
        "expected_head": state.metadata[PUBLICATION_KEY]["head_id"],
        "previous_model": model,
        "training_generation": training_generation,
        "accepted_training": accepted,
        "candidate": candidate,
        "config_sha256": state.metadata["bootstrap_config_hash"],
        "effective_runtime": state.metadata.get(
            BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY
        ),
        "policy": dict(POLICY),
        "expected_cursor_sha256": cursor_digest(transaction["accepted_cursor"]),
        "receipt_ledger_sha256": object_digest(state.metadata[RECEIPTS_METADATA_KEY]),
    }


def prepare_model_activation_request(
    paths: MorpionBootstrapPaths, *, training_generation: int
) -> dict[str, Any]:
    """Read-only capture for an explicit caller decision; Training never invokes this."""
    with growth_receipt_ownership(paths):
        return _request(
            paths, load_bootstrap_run_state(paths.run_state_path), training_generation
        )


def _save(
    session: GrowthPatchReceiptSession,
    runner: AnemoneMorpionSearchRunner,
    state: MorpionBootstrapRunState,
) -> MorpionBootstrapRunState:
    generation = state.generation + 1
    assert state.latest_runtime_checkpoint_path is not None
    checkpoint = session.checkpoint_path(
        generation,
        "sharded"
        if inside(session.paths.work_dir, state.latest_runtime_checkpoint_path).is_dir()
        else "json-zst",
    )
    runner.save_checkpoint(checkpoint)
    science = session.export_scientific(generation)
    assert session.publication is not None
    model = session.publication.active_evaluator
    assert model is not None
    return session.commit(
        state,
        replace(
            state,
            generation=generation,
            last_save_unix_s=time.time(),
            tree_size_at_last_save=runner.current_tree_size(),
            latest_runtime_checkpoint_path=session.paths.relative_to_work_dir(
                checkpoint
            ),
            latest_tree_snapshot_path=session.paths.relative_to_work_dir(science),
            active_evaluator_name=model["name"],
        ),
    )


def activation_completion(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Verify publication-v3/request-v1 completion, not historical scientific data.

    The atomic authority binds the exact sealed descriptor and request digest.
    Its retained metadata proves the committed transition even when old payloads
    are unavailable. It does not authorize reading, restoring or deleting them.
    A coherent malicious rewrite of the authority is outside its existing trust
    boundary; this is a hash commitment, not a cryptographic signature.
    """
    records = publication_records(state.metadata)
    require(record in records, "Activation completion is not committed.")
    index = records.index(record)
    require(
        index > 0
        and record.get("publication_version") == 3
        and state.metadata[PUBLICATION_KEY].get("evaluator_contract") == 1,
        "Unsupported activation completion contract.",
    )
    descriptor = read_committed_descriptor(paths.work_dir, record)
    root = capsule_path(paths.work_dir, record["id"], 3).relative_to(paths.work_dir)
    checkpoint_roots = {
        "/".join(entry["path"].split("/")[:2])
        for entry in descriptor["files"]
        if entry["path"].startswith("checkpoint/")
    }
    require(
        len(checkpoint_roots) == 1
        and record["checkpoint_path"]
        == f"{root.as_posix()}/{next(iter(checkpoint_roots))}"
        and record["tree_snapshot_path"]
        == f"{root.as_posix()}/{descriptor['sharded_snapshot_path']}",
        "Activation completion payload pointer mismatch.",
    )
    completion = descriptor.get("model_activation")
    require(isinstance(completion, dict), "Missing activation completion evidence.")
    assert isinstance(completion, dict)
    request = completion.get("request")
    require(isinstance(request, dict), "Missing completed activation request.")
    assert isinstance(request, dict)
    require(
        isinstance(request.get("schema_version"), int)
        and not isinstance(request["schema_version"], bool)
        and request["schema_version"] == 1
        and completion.get("request_sha256")
        == record.get("activation_request_sha256")
        == object_digest(request)
        and request.get("identity") == workspace_epoch(state)
        and object_digest(request.get("policy")) == object_digest(POLICY)
        and request.get("config_sha256")
        == descriptor["bootstrap_config_hash"]
        == state.metadata.get("bootstrap_config_hash"),
        "Activation completion request binding mismatch.",
    )
    parent = records[index - 1]
    producer = parent["scientific_identity"]["producer"]["identity"]
    require(
        completion.get("previous_head") == record["parent_id"] == parent["id"]
        and completion.get("previous_producer") == producer
        and request["previous_model"]["sha256"] == producer["evaluator"]
        and descriptor.get("active_evaluator") == request["candidate"]
        and descriptor["scientific_identity"]["producer"]["identity"]
        == {**producer, "evaluator": request["candidate"]["sha256"]}
        and request["expected_head"]
        in {r["id"] for r in records[max(0, index - 2) : index]}
        and any(
            r["generation"] == request["training_generation"]
            and r.get("accepted_training") == request["accepted_training"]
            for r in records[:index]
        )
        and isinstance(completion.get("refresh"), dict),
        "Activation completion transition mismatch.",
    )
    return completion


def activate_model(
    paths: MorpionBootstrapPaths, request: dict[str, Any]
) -> MorpionBootstrapRunState:
    """Explicit M1→M2 transition; old state remains authoritative until ordinary commit."""
    request = copy.deepcopy(request)
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        digest = object_digest(request)
        records = publication_records(state.metadata)
        matching = [r for r in records if r.get("activation_request_sha256") == digest]
        require(len(matching) <= 1, "Duplicate committed activation request.")
        if matching:
            completion = activation_completion(paths, state, matching[0])
            require(
                completion["request"] == request,
                "Activation receipt mismatch.",
            )
            # Exact retry repairs only projections; it never reactivates an old M2.
            producer = publication_records(state.metadata)[-1]["scientific_identity"][
                "producer"
            ]["identity"]
            repair = GrowthPatchReceiptSession(
                paths,
                object(),
                durable_generation_publication=True,
                scientific_v2_producer=producer,
                closed_scientific_snapshot=True,
            )
            assert repair.publication is not None
            try:
                repair.publication.prepare(state)
                repair.publish_manifest(state)
                repair.acknowledge_pending(state)
            finally:
                repair.publication.close()
            return state
        require(
            digest
            == object_digest(_request(paths, state, request["training_generation"])),
            "Stale or conflicting model activation request.",
        )
        measurement = StageMeasurement()
        runner, effective = _runner(paths, state)
        head = publication_records(state.metadata)[-1]
        producer = dict(head["scientific_identity"]["producer"]["identity"])
        session = GrowthPatchReceiptSession(
            paths,
            runner,
            durable_generation_publication=True,
            scientific_v2_producer=producer,
            closed_scientific_snapshot=True,
        )
        publication = session.publication
        assert publication is not None
        try:
            publication.prepare(state)
            session.validate_state(state)
            session.begin_cycle(None)
            assert state.latest_runtime_checkpoint_path is not None
            with torch.random.fork_rng(devices=[]):
                runner.load_or_create(
                    inside(paths.work_dir, state.latest_runtime_checkpoint_path),
                    session.runtime_model_path(None),
                    effective,
                )
            runner.validate_model_refresh_certificates()
            # No nested lease: the existing session methods require caller ownership.
            result = session.apply(state, effective)
            if result.patch_applied:
                state = _save(session, runner, state)
                session.begin_cycle(None)
                assert state.latest_runtime_checkpoint_path is not None
                with torch.random.fork_rng(devices=[]):
                    runner.load_or_create(
                        inside(paths.work_dir, state.latest_runtime_checkpoint_path),
                        session.runtime_model_path(None),
                        effective,
                    )
                runner.validate_model_refresh_certificates()
            require(
                not paths.pipeline_reevaluation_patch_path.exists(),
                "Pending proposal was not durably acknowledged.",
            )
            model = request["candidate"]
            model_path = publication.freeze_model(model)
            refresh_measurement = StageMeasurement()
            refresh = runner.refresh_model_estimates(model_path)
            publication.activation = {
                "request_sha256": digest,
                "request": request,
                "previous_head": state.metadata[PUBLICATION_KEY]["head_id"],
                "previous_producer": producer,
                "refresh": asdict(refresh),
                "refresh_performance": refresh_measurement.finish(),
            }
            publication.active_evaluator = model
            publication.producer = {**producer, "evaluator": model["sha256"]}
            state = _save(session, runner, state)
            persist_stage_measurement(
                paths.work_dir, state.generation, "activation", measurement.finish()
            )
            return state
        finally:
            session.discard_runtime()
            publication.close()


def activate_selected_round_automatically(paths: MorpionBootstrapPaths) -> None:
    """Opt-in Growth boundary: persist and retry the ordinary atomic request exactly."""
    from .derived.provenance import atomic_json
    from .learning_policy import policy_from_metadata
    from .training_publication import ACCEPTED, _proposal

    config = load_bootstrap_config(paths.bootstrap_config_path)
    policy = policy_from_metadata(config.metadata)
    require(
        policy is not None and policy.automatic_activation,
        "Automatic activation requires explicit persisted experiment authorization.",
    )
    state = load_bootstrap_run_state(paths.run_state_path)
    if state.metadata.get("experiment_termination") is not None:
        return
    directory = paths.pipeline_dir / "automatic_activation"
    # Preserve original requests and any explicit M1-drain renewals. A committed
    # member finishes its entire chain; no old request is reactivated afterward.
    chains: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("generation_*.json")):
        request = read_object(path)
        chains.setdefault(request["training_generation"], []).append(request)
    positions = {r["id"]: i for i, r in enumerate(publication_records(state.metadata))}
    for requests in chains.values():
        committed = {
            r.get("activation_request_sha256")
            for r in publication_records(state.metadata)
        }
        if any(object_digest(r) in committed for r in requests):
            continue
        require(
            all(r["expected_head"] in positions for r in requests),
            "Unknown automatic request head.",
        )
        request = max(requests, key=lambda r: positions[r["expected_head"]])
        if request["expected_head"] != state.metadata[PUBLICATION_KEY]["head_id"]:
            with growth_receipt_ownership(paths):
                state = load_bootstrap_run_state(paths.run_state_path)
                request = _renew_automatic_request_after_drain(paths, state, request)
                renewed_path = directory / (
                    f"generation_{request['training_generation']:06d}"
                    f".recovery_{request['expected_head']}.json"
                )
                if renewed_path.exists():
                    require(
                        read_object(renewed_path) == request,
                        "Conflicting automatic renewal.",
                    )
                else:
                    atomic_json(renewed_path, request)
        state = activate_model(paths, request)
    accepted = [r for r in publication_records(state.metadata) if ACCEPTED in r]
    if not accepted:
        return
    latest = accepted[-1]
    proposal = _proposal(paths, latest[ACCEPTED])
    decision = proposal["status"]["metadata"].get("continual_selection")
    require(
        isinstance(decision, dict),
        "Automatic activation requires a continual selection decision.",
    )
    if not decision["activation_eligible"]:
        return
    path = directory / f"generation_{latest['generation']:06d}.json"
    if path.exists():
        return  # The exact request above was already committed or recovered.
    if committed_evaluator(paths.work_dir, state)[0] != decision["incumbent"]["model"]:
        # A learner may finish on the M1 receipt-drain head while an interrupted
        # M1->M2 request is recovered. Its old-incumbent comparison cannot justify
        # another activation. Keep M2 and let ordinary Growth/Dataset continue.
        LOGGER.info(
            "[activation] selection_superseded generation=%s reason=incumbent_changed",
            latest["generation"],
        )
        return
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        accepted_now = [r for r in publication_records(state.metadata) if ACCEPTED in r]
        require(
            accepted_now[-1][ACCEPTED] == latest[ACCEPTED],
            "A newer selection superseded this round.",
        )
        if path.exists():
            return
        request = _request(paths, state, latest["generation"])
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(path, request)
    activate_model(paths, request)


def _renew_automatic_request_after_drain(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Use the existing new-request contract only after one unchanged M1 drain.

    Caller owns Growth. No candidate, selection, policy or runtime is retargeted;
    the original file remains evidence. Arbitrary stale requests still fail.
    """
    from .sharded_training_reader import ShardedTrainingGenerationView

    records = publication_records(state.metadata)
    require(
        len(records) >= 2
        and records[-2]["id"] == request["expected_head"]
        and records[-1]["parent_id"] == request["expected_head"]
        and "activation_request_sha256" not in records[-1]
        and records[-1]["scientific_identity"]["producer"]
        == records[-2]["scientific_identity"]["producer"]
        and any(
            r.committed_generation == state.generation
            for r in receipt_ledger(state).values()
        ),
        "Stale automatic request is not an adjacent M1 receipt drain.",
    )
    renewed = _request(paths, state, request["training_generation"])
    changed = {"expected_head", "expected_cursor_sha256", "receipt_ledger_sha256"}
    require(
        {k: v for k, v in renewed.items() if k not in changed}
        == {k: v for k, v in request.items() if k not in changed},
        "Automatic renewal would retarget the accepted candidate or policy.",
    )
    # A Growth publication can also accept a numeric receipt. Compare all static
    # scientific fields and visit counts, so such a head cannot masquerade as a
    # receipt-only boundary. Owned disposable indices keep this rare audit bounded.
    with (
        TemporaryDirectory(prefix="morpion-activation-recovery-") as scratch,
        ShardedTrainingGenerationView(
            inside(paths.work_dir, records[-2]["tree_snapshot_path"]),
            cache_dir=Path(scratch) / "old",
            allow_closed_snapshot=True,
        ) as old,
        ShardedTrainingGenerationView(
            inside(paths.work_dir, records[-1]["tree_snapshot_path"]),
            cache_dir=Path(scratch) / "new",
            allow_closed_snapshot=True,
        ) as new,
    ):
        for left, right in zip_longest(
            old.iter_nodes_in_export_order(), new.iter_nodes_in_export_order()
        ):
            require(
                left is not None and right is not None,
                "Growth changed the activation tree.",
            )
            assert left is not None and right is not None
            static = (
                "node_id",
                "parent_ids",
                "child_ids",
                "depth",
                "state_ref_payload",
                "is_terminal",
                "visit_count",
            )
            require(
                all(getattr(left, field) == getattr(right, field) for field in static),
                "Growth changed the activation tree.",
            )
    return renewed
