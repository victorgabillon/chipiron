"""Explicit accepted-model activation under the existing Growth commit authority.

Only fresh format-3/default-max/Linoo workspaces opt in at genesis. The request is
an optimistic binding, not a journal. Pending M1 work commits first; activation
then owns one private runtime and publishes through the ordinary capsule/receipt
session. Failure discards RAM. Recovery reads the existing authoritative head.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, cast

import torch
from anemone.node_evaluation.evaluator_refresh import validate_refresh_node
from valanga.evaluations import Certainty

from chipiron.environments.morpion.learning.target_confidence import (
    validate_confidence_policy,
)
from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionStateEvaluator,
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
    committed_evaluator,
    growth_receipt_ownership,
    object_digest,
    publication_records,
    require,
    validate_capsule,
)
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
    from collections.abc import Iterable, Mapping

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

    from chipiron.environments.morpion.types import MorpionState

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


def _certificates(runner: AnemoneMorpionSearchRunner) -> None:
    """Verify terminal facts/scores independently; reuse Anemone's tree proof rule."""
    evaluator = MorpionStateEvaluator()
    for node in cast(
        "Iterable[AlgorithmNode[MorpionState]]", runner.iter_profile_nodes()
    ):
        direct = node.tree_evaluation.direct_value
        # The state, not a saved exactness bit, decides terminality.
        terminal = node.state.is_game_over()
        if terminal or (direct is not None and direct.certainty == Certainty.TERMINAL):
            require(
                terminal and direct == evaluator.evaluate(node.state),
                "Terminal certificate disagrees with the actual state or terminal score.",
            )
        validate_refresh_node(node, None)  # Unknown direct FORCED fails closed.


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
        _certificates(runner)
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


def activate_model(
    paths: MorpionBootstrapPaths, request: dict[str, Any]
) -> MorpionBootstrapRunState:
    """Explicit M1→M2 transition; old state remains authoritative until ordinary commit."""
    request = copy.deepcopy(request)
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        digest = object_digest(request)
        for record in publication_records(state.metadata):
            if record.get("activation_request_sha256") == digest:
                descriptor = validate_capsule(paths.work_dir, record)
                require(
                    descriptor["model_activation"]["request"] == request,
                    "Activation receipt mismatch.",
                )
                # Exact retry repairs only projections; it never reactivates an old M2.
                producer = publication_records(state.metadata)[-1][
                    "scientific_identity"
                ]["producer"]["identity"]
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
            _certificates(runner)
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
                _certificates(runner)
            require(
                not paths.pipeline_reevaluation_patch_path.exists(),
                "Pending proposal was not durably acknowledged.",
            )
            model = request["candidate"]
            model_path = publication.freeze_model(model)
            refresh = runner.refresh_model_estimates(model_path)
            publication.activation = {
                "request_sha256": digest,
                "request": request,
                "previous_head": state.metadata[PUBLICATION_KEY]["head_id"],
                "previous_producer": producer,
                "refresh": asdict(refresh),
            }
            publication.active_evaluator = model
            publication.producer = {**producer, "evaluator": model["sha256"]}
            return _save(session, runner, state)
        finally:
            session.discard_runtime()
            publication.close()
