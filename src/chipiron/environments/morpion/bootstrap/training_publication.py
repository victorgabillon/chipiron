"""Accept immutable Training results without activating their models in Growth."""

import json
import os
import platform
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from . import dataset_publication as dataset
from .derived.provenance import atomic_json, inside
from .generation_publication_io import (
    artifact_digest,
    copy_artifact,
    file_digest,
    generation_read_pin,
    growth_receipt_ownership,
    publish_directory_if_absent,
    read_object,
    require,
    sync_artifact,
)
from .pipeline_artifacts import (
    MorpionPipelineGenerationManifest,
    load_pipeline_manifest,
    load_pipeline_stage_claim,
    pipeline_manifest_to_dict,
    pipeline_training_status_from_dict,
    pipeline_training_status_to_dict,
)
from .pipeline_claims import claim_pipeline_stage, release_owned_pipeline_stage_claim
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state
from .worker_result_io import (
    attempt_directory,
    publication_record,
    read_proposal,
    result_protocol_enabled,
    with_accepted_result,
    workspace_epoch,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .control import MorpionBootstrapControl
    from .evaluator_config import MorpionEvaluatorsConfig
    from .pipeline_artifacts import MorpionPipelineTrainingStatusArtifact
    from .run_state import MorpionBootstrapRunState

KEY = "durable_training"
ACCEPTED = "accepted_training"
NAMESPACE = "training_publications_v1"
CONTRACT = {"schema_version": 1}


def enabled(state: MorpionBootstrapRunState) -> bool:
    """Only explicit fresh genesis enables this protocol; downgrades fail closed."""
    if not result_protocol_enabled(state, KEY, ACCEPTED):
        return False
    require(dataset.enabled(state), "Training requires fenced Dataset.")
    return True


def configuration(
    args: MorpionBootstrapArgs,
    evaluators: MorpionEvaluatorsConfig,
    control: MorpionBootstrapControl,
) -> dict[str, Any]:
    """Capture effective trainer arguments, including defaults and candidate order."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.model_args import (
        morpion_regressor_args_from_training_args,
    )

    from .cycle_training import morpion_training_args_from_evaluator_spec

    if args.continual_learning is not None:
        require(
            args.training_evaluator_names is None,
            "Continual rounds train every configured candidate.",
        )
        require(
            control.force_evaluator is None,
            "Continual selection requires measured validation loss.",
        )
        require(
            args.validation_fraction > 0,
            "Continual training requires held-out validation.",
        )
    candidates = []
    for name, spec in evaluators.evaluators.items():
        require(
            name not in {"", ".", ".."} and "/" not in name and "\\" not in name,
            "Unsafe evaluator output name.",
        )
        trainer = morpion_training_args_from_evaluator_spec(
            spec=spec,
            dataset_file="accepted Dataset",
            output_dir=name,
            shuffle=args.shuffle,
            validation_fraction=args.validation_fraction,
            validation_seed=args.validation_seed,
            device=args.training_device,
        )
        if args.continual_learning is not None:
            trainer = replace(trainer, stable_validation_split=True)
        candidates.append({
            "name": name,
            "args": asdict(trainer),
            "model_args": asdict(morpion_regressor_args_from_training_args(trainer)),
        })
    return cast(
        "dict[str, Any]",
        json.loads(
            json.dumps({
                "candidates": candidates,
                "force_evaluator": control.force_evaluator,
                "selection": "heldout_candidate_name_tie_incumbent_on_equal_v1"
                if args.continual_learning is not None
                else "forced_else_lowest_finite_validation_else_final_v1",
                "max_rows": args.training_max_rows,
                "chunk_size": args.training_row_chunk_size,
                "skip_diagnostics": args.skip_evaluator_diagnostics,
                "diagnostics_max_rows": args.evaluator_diagnostics_max_rows,
                **(
                    {"continual_learning": asdict(args.continual_learning)}
                    if args.continual_learning is not None
                    else {}
                ),
            })
        ),
    )


def reproducibility() -> dict[str, Any]:
    """Observe existing RNG/settings without reseeding or changing numerical behavior."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_initial_seed": torch.initial_seed(),
        "torch_rng_hex": bytes(torch.get_rng_state().tolist()).hex(),
        "cuda_rng_hex": [
            bytes(r.tolist()).hex() for r in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else [],
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": torch.get_num_threads(),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("MORPION_", "OMP_", "MKL_", "CUBLAS_", "CUDA_"))
        },
    }


@dataclass(frozen=True)
class TrainingAttempt:
    """One captured input and private computation directory, never an active model."""

    admission: dict[str, Any]
    directory: Path


def latest_candidate_parents(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    names: list[str],
) -> dict[str, Any]:
    """Derive candidate heads from existing accepted receipts, never model projections."""
    from .generation_publication_io import publication_records

    parents: dict[str, Any] = dict.fromkeys(names)
    unresolved = set(names)
    for record in reversed(publication_records(state.metadata)):
        if ACCEPTED not in record:
            continue
        proposal = _proposal(paths, record[ACCEPTED])
        for name in sorted(unresolved & proposal["models"].keys()):
            model = proposal["models"][name]
            require(
                artifact_digest(inside(paths.work_dir, model["path"]))
                == model["sha256"],
                "Latest accepted candidate bundle is missing or changed.",
            )
            parents[name] = {
                "generation": record["generation"],
                "accepted_training": record[ACCEPTED],
                "model": model,
                "weights_sha256": file_digest(
                    inside(paths.work_dir, model["path"]) / "param.pt"
                ),
            }
            unresolved.remove(name)
        if not unresolved:
            break
    return parents


def _continuation_admission(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    config: dict[str, Any],
    generation: int,
) -> dict[str, Any]:
    """Capture parent and incumbent identities under the existing ownership fence."""
    from .generation_publication_io import committed_evaluator, publication_records

    if "continual_learning" not in config:
        return {}
    records = publication_records(state.metadata)
    parents = latest_candidate_parents(
        paths, state, [c["name"] for c in config["candidates"]]
    )
    require(
        all(p is None or p["generation"] < generation for p in parents.values()),
        "A newer accepted candidate cannot be trained on an older generation.",
    )
    accepted = [r for r in records if ACCEPTED in r]
    previous_generation = max((r["generation"] for r in accepted), default=0)
    incumbent, _ = committed_evaluator(paths.work_dir, state)
    return {
        "parents": parents,
        "round_number": len(accepted) + 1,
        "incumbent": incumbent,
        "skipped_scientific_generations": [
            r["generation"]
            for r in records
            if previous_generation < r["generation"] < generation
        ],
        "skipped_accepted_dataset_generations": [
            r["generation"]
            for r in records
            if previous_generation < r["generation"] < generation
            and dataset.ACCEPTED in r
            and ACCEPTED not in r
        ],
    }


def _directory(
    paths: MorpionBootstrapPaths, identifier: str, *, private: bool = False
) -> Path:
    return attempt_directory(paths, identifier, namespace=NAMESPACE, private=private)


def _proposal(
    paths: MorpionBootstrapPaths, reference: dict[str, Any]
) -> dict[str, Any]:
    return read_proposal(paths, reference, namespace=NAMESPACE)


@contextmanager
def admit_training(
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    config: dict[str, Any],
    *,
    ttl_seconds: float,
    owner: str | None,
) -> Iterator[TrainingAttempt | MorpionPipelineGenerationManifest]:
    """Capture under Growth ownership; caller retains scientific pin during computation."""
    generation = manifest.generation
    claim_path = paths.pipeline_training_claim_path_for_generation(generation)
    attempt = None
    result: TrainingAttempt | MorpionPipelineGenerationManifest
    with growth_receipt_ownership(paths, wait=True):
        state = load_bootstrap_run_state(paths.run_state_path)
        require(enabled(state), "Durable Training is not enabled.")
        from .learning_policy import policy_from_metadata, validate_continual_context

        validate_continual_context(paths, state, policy_from_metadata(config))
        record = publication_record(state, generation)
        if ACCEPTED in record:
            proposal = _proposal(paths, record[ACCEPTED])
            require(
                proposal["admission"]["configuration"] == config,
                "Conflicting accepted Training configuration.",
            )
            result = project_training(paths, state, manifest, verify=True)
            _repair(paths, result, proposal)
            release_owned_pipeline_stage_claim(
                claim_path=claim_path, claim_id=proposal["admission"]["claim_id"]
            )
        else:
            from .pipeline.cursors import training_lower_bound_details

            bound = training_lower_bound_details(paths)
            # Active claims are scheduling guards, not evidence that a process died.
            if generation <= (bound.cursor_completed_generation or 0) or (
                generation <= bound.generation
                and generation != bound.cursor_started_generation
            ):
                result = manifest
            else:
                source = dataset.accepted_dataset_identity(paths, state, generation)
                claim = claim_pipeline_stage(
                    generation=generation,
                    stage="training",
                    claim_path=claim_path,
                    ttl_seconds=ttl_seconds,
                    owner=owner,
                )
                identifier = uuid.uuid4().hex
                staging = _directory(paths, identifier, private=True)
                staging.mkdir(parents=True, exist_ok=False)
                admission = {
                    "schema_version": 2 if "continual_learning" in config else 1,
                    "attempt_id": identifier,
                    "generation": generation,
                    "identity": workspace_epoch(state),
                    "dataset": source,
                    "claim_id": claim.claim_id,
                    "configuration": config,
                    "reproducibility": reproducibility(),
                    **_continuation_admission(paths, state, config, generation),
                }
                atomic_json(staging / "admission.json", admission)
                attempt = TrainingAttempt(admission, staging)
                result = attempt
    try:
        yield result
    finally:
        if attempt is not None:
            try:
                with growth_receipt_ownership(paths, wait=True):
                    release_owned_pipeline_stage_claim(
                        claim_path=claim_path, claim_id=attempt.admission["claim_id"]
                    )
            except BlockingIOError:
                pass


def _validate(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    proposal: dict[str, Any],
    *,
    verify: bool,
    record: dict[str, Any] | None = None,
) -> MorpionPipelineTrainingStatusArtifact:
    require(enabled(state), "Training authority is required.")
    require(
        set(proposal) == {"admission", "models", "status"},
        "Incomplete Training result.",
    )
    admission = proposal["admission"]
    from .learning_policy import policy_from_metadata, validate_continual_context

    validate_continual_context(
        paths, state, policy_from_metadata(admission["configuration"])
    )
    require(
        set(admission)
        == {
            "schema_version",
            "attempt_id",
            "generation",
            "identity",
            "dataset",
            "claim_id",
            "configuration",
            "reproducibility",
        }
        | (
            {
                "parents",
                "round_number",
                "incumbent",
                "skipped_accepted_dataset_generations",
                "skipped_scientific_generations",
            }
            if "continual_learning" in admission["configuration"]
            else set()
        )
        and isinstance(admission["schema_version"], int)
        and not isinstance(admission["schema_version"], bool)
        and admission["schema_version"]
        == (2 if "continual_learning" in admission["configuration"] else 1),
        "Incomplete Training admission.",
    )
    require(
        admission["identity"] == workspace_epoch(state),
        "Training workspace/epoch mismatch.",
    )
    generation = admission["generation"]
    record = publication_record(state, generation) if record is None else record
    require(
        record.get(dataset.ACCEPTED) == admission["dataset"]["reference"],
        "Accepted Dataset changed.",
    )
    require(
        admission["dataset"]["source"]
        == {k: record[k] for k in ("id", "descriptor_sha256", "scientific_identity")},
        "Training scientific source changed.",
    )
    root = _directory(paths, admission["attempt_id"])
    require(
        read_object(inside(root, "admission.json")) == admission,
        "Training admission changed.",
    )
    status = pipeline_training_status_from_dict(proposal["status"])
    names = [c["name"] for c in admission["configuration"]["candidates"]]
    require(
        len(names) == len(set(names))
        and set(names) == set(status.evaluator_results) == set(proposal["models"])
        and status.generation == generation
        and status.status == "done"
        and status.selected_evaluator_name in names,
        "Incomplete Training candidate outputs.",
    )
    from .cycle_training import (
        _has_validation_loss,
        select_or_force_active_evaluator_name,
    )
    from .history import MorpionEvaluatorMetrics

    metrics = {
        name: MorpionEvaluatorMetrics(
            final_loss=r.final_loss,
            validation_loss=r.validation_loss,
            num_epochs=r.num_epochs,
            num_samples=None,
            num_validation_samples=r.num_validation_samples,
        )
        for name in names
        for r in [status.evaluator_results[name]]
    }
    require(
        select_or_force_active_evaluator_name(
            evaluator_metrics=metrics,
            force_evaluator=admission["configuration"]["force_evaluator"],
            validation_only="continual_learning" in admission["configuration"],
        )
        == status.selected_evaluator_name,
        "Training selection changed.",
    )
    expected_policy = (
        "forced_evaluator"
        if admission["configuration"]["force_evaluator"] is not None
        else (
            "lowest_validation_loss"
            if _has_validation_loss(metrics)
            else "lowest_final_loss"
        )
    )
    require(
        status.selection_policy == expected_policy, "Training selection policy changed."
    )
    if "parents" in admission:
        _validate_continual_selection(admission, status)
    for name in names:
        model = proposal["models"][name]
        expected = paths.relative_to_work_dir(
            root / "models" / f"generation_{generation:06d}" / name
        )
        require(
            model["path"]
            == status.evaluator_results[name].model_bundle_path
            == expected,
            "Training model path mismatch.",
        )
        if verify:
            require(
                artifact_digest(inside(paths.work_dir, expected)) == model["sha256"],
                "Training model missing or changed.",
            )
            if "parents" in admission:
                _validate_continuation(paths, admission, name, model)
    if verify:
        require(
            dataset.accepted_dataset_identity(paths, state, generation)
            == admission["dataset"],
            "Training Dataset bytes changed.",
        )
    return status


def _validate_continual_selection(
    admission: dict[str, Any], status: MorpionPipelineTrainingStatusArtifact
) -> None:
    """Independently reapply the declared decision rule to same-round held-out metrics."""
    import math

    decision = status.metadata.get("continual_selection")
    require(isinstance(decision, dict), "Missing continual selection decision.")
    assert isinstance(decision, dict)
    losses = {
        name: result.validation_loss
        for name, result in status.evaluator_results.items()
    }
    require(
        all(loss is not None and math.isfinite(loss) for loss in losses.values()),
        "Every candidate requires finite held-out loss.",
    )
    winner = status.selected_evaluator_name
    assert winner is not None
    incumbent = decision["incumbent"]
    require(
        incumbent["model"] == admission["incumbent"]
        and math.isfinite(incumbent["validation_loss"]),
        "Selection incumbent or its validation metric changed.",
    )
    winning_loss = losses[winner]
    assert winning_loss is not None
    improves = winning_loss < incumbent["validation_loss"]
    policy = admission["configuration"]["continual_learning"]
    number = admission["round_number"]
    eligible = (
        number % policy["selection_every_rounds"] == 0
        and number % policy["activation_every_rounds"] == 0
    )
    population = decision["validation_population"]
    require(
        decision["round_number"] == number
        and decision["candidate"] == winner
        and decision["candidate_validation_losses"] == losses
        and decision["improves_incumbent"] == improves
        and decision["activation_eligible"] == (eligible and improves)
        and decision["decision"]
        == ("candidate" if eligible and improves else "retain_incumbent")
        and population["validation_count"] > 0
        and all(
            r.num_validation_samples == population["validation_count"]
            and r.num_train_samples == population["train_count"]
            for r in status.evaluator_results.values()
        ),
        "Continual selection does not match the shared held-out decision contract.",
    )


def _validate_continuation(
    paths: MorpionBootstrapPaths,
    admission: dict[str, Any],
    name: str,
    model: dict[str, Any],
) -> None:
    """Check exact loaded-parent evidence in the accepted immutable model bundle."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        load_morpion_model_bundle,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
        candidate_training_seed,
        weights_digest,
    )

    with torch.random.fork_rng(devices=[]):
        trained, _, manifest = load_morpion_model_bundle(
            inside(paths.work_dir, model["path"])
        )
    evidence = manifest.metadata["continuation"]
    parent = admission["parents"][name]
    policy = admission["configuration"]["continual_learning"]
    inherits = (
        parent is not None
        and policy["training_policy"] == "continue_from_previous_accepted_weights"
    )
    require(
        evidence["candidate_name"] == name
        and evidence["training_policy"] == policy["training_policy"]
        and evidence["optimizer_state_policy"] == policy["optimizer_state_policy"]
        and evidence["loaded_previous_weights"] == inherits
        and evidence["parent_bundle_sha256"]
        == (None if parent is None else parent["model"]["sha256"])
        and evidence["parent_weights_sha256"]
        == (None if parent is None else parent["weights_sha256"])
        and evidence["final_weights_sha256"] == weights_digest(trained)
        and evidence["optimizer_steps"] > 0,
        "Training continuation evidence differs from admitted lineage.",
    )
    require(
        evidence["seed"]
        == candidate_training_seed(policy["seed"], name, admission["generation"])
        and evidence["optimizer_state_loaded"]
        == (inherits and policy["optimizer_state_policy"] == "continue_adam")
        and evidence["optimizer_sha256"]
        == file_digest(inside(paths.work_dir, model["path"]) / "optimizer.pt")
        and evidence["final_optimizer_step_count"]
        - evidence["starting_optimizer_step_count"]
        == evidence["optimizer_steps"]
        and evidence["training_examples_seen"]
        == manifest.metadata["num_train_samples"] * manifest.metadata["num_epochs"],
        "Candidate optimizer, seed or sample provenance changed.",
    )
    if inherits:
        assert parent is not None
        bundle = inside(paths.work_dir, parent["model"]["path"])
        require(
            artifact_digest(bundle) == parent["model"]["sha256"],
            "Accepted parent changed during training.",
        )
        with torch.random.fork_rng(devices=[]):
            previous, _, _ = load_morpion_model_bundle(bundle)
        require(
            evidence["starting_weights_sha256"] == weights_digest(previous),
            "Training did not start with the exact accepted parent tensors.",
        )


def accepted_selected_model(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    generation: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the accepted result and all its original Dataset/model dependencies."""
    record = publication_record(state, generation)
    reference = record.get(ACCEPTED)
    require(reference is not None, "No exactly accepted Training result.")
    assert isinstance(reference, dict)
    proposal = _proposal(paths, reference)
    status = _validate(paths, state, proposal, verify=True, record=record)
    name = status.selected_evaluator_name
    assert name is not None
    result = status.evaluator_results[name]
    require(
        result.num_train_samples is not None
        and result.num_train_samples > 0
        and result.num_epochs is not None
        and result.num_epochs > 0,
        "Activation requires a genuinely trained candidate with positive samples.",
    )
    return reference, {
        **proposal["models"][name],
        "generation": generation,
        "name": name,
        "source": "local_training",
        "source_generation": generation,
        "local_trained_generation": generation,
    }


def project_training(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    manifest: MorpionPipelineGenerationManifest,
    *,
    verify: bool = False,
    record: dict[str, Any] | None = None,
) -> MorpionPipelineGenerationManifest:
    """Derive completion from authority; visible model files and statuses prove nothing."""
    record = (
        publication_record(state, manifest.generation) if record is None else record
    )
    reference = record.get(ACCEPTED)
    if reference is None:
        return replace(
            manifest,
            training_status="not_started",
            selected_evaluator_name=None,
            model_bundle_paths={},
        )
    status = _validate(
        paths, state, _proposal(paths, reference), verify=verify, record=record
    )
    if verify:
        sync_artifact(paths.run_state_path, paths.work_dir)
    return replace(
        manifest,
        training_status="done",
        selected_evaluator_name=status.selected_evaluator_name,
        model_bundle_paths={
            name: r.model_bundle_path for name, r in status.evaluator_results.items()
        },
        metadata={**manifest.metadata, **status.metadata},
    )


def publish_training(
    paths: MorpionBootstrapPaths,
    attempt: TrainingAttempt,
    status: MorpionPipelineTrainingStatusArtifact,
) -> dict[str, Any]:
    """Seal existing model bundles, then publish through the shared no-replace primitive."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        load_morpion_model_bundle,
    )

    require(
        read_object(attempt.directory / "admission.json") == attempt.admission,
        "Training admission changed.",
    )
    require(
        bool(status.evaluator_results)
        and all(
            result.num_train_samples is not None
            and result.num_train_samples > 0
            and result.num_epochs is not None
            and result.num_epochs > 0
            for result in status.evaluator_results.values()
        ),
        "Training publication requires positive effective samples and completed epochs.",
    )
    final = _directory(paths, attempt.admission["attempt_id"])
    models = {}
    results = {}
    for name, result in status.evaluator_results.items():
        relative = f"models/generation_{status.generation:06d}/{name}"
        bundle = inside(attempt.directory, relative)
        # The normal loader validates architecture/weight shapes. Preserve caller RNG.
        with torch.random.fork_rng(devices=[]):
            model, model_args, model_manifest = load_morpion_model_bundle(bundle)
            del model
        candidate = next(
            c
            for c in attempt.admission["configuration"]["candidates"]
            if c["name"] == name
        )
        require(
            json.loads(json.dumps(asdict(model_args))) == candidate["model_args"],
            "Training architecture changed.",
        )
        require(
            all(
                model_manifest.metadata[key] == getattr(result, key)
                for key in (
                    "final_loss",
                    "train_loss",
                    "validation_loss",
                    "num_epochs",
                    "batch_size",
                    "learning_rate",
                )
            ),
            "Training model metrics changed.",
        )
        require(
            all(
                model_manifest.metadata[key] == candidate["args"][key]
                for key in (
                    "shuffle",
                    "validation_fraction",
                    "validation_seed",
                    "num_epochs",
                    "batch_size",
                    "learning_rate",
                )
            ),
            "Training hyperparameters changed.",
        )
        target = paths.relative_to_work_dir(final / relative)
        models[name] = {"path": target, "sha256": artifact_digest(bundle)}
        results[name] = replace(result, model_bundle_path=target)
    status = replace(status, evaluator_results=results)
    proposal = {
        "admission": attempt.admission,
        "models": models,
        "status": pipeline_training_status_to_dict(status),
    }
    atomic_json(attempt.directory / "proposal.json", proposal)
    sync_artifact(attempt.directory, paths.work_dir)
    final.parent.mkdir(parents=True, exist_ok=True)
    publish_directory_if_absent(attempt.directory, final, paths.work_dir)
    return {
        "attempt_id": attempt.admission["attempt_id"],
        "descriptor_sha256": file_digest(final / "proposal.json"),
    }


def _repair(
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    proposal: dict[str, Any],
) -> None:
    atomic_json(
        paths.pipeline_manifest_path_for_generation(manifest.generation),
        pipeline_manifest_to_dict(manifest),
    )
    atomic_json(
        paths.pipeline_training_status_path_for_generation(manifest.generation),
        proposal["status"],
    )


def accept_training(
    paths: MorpionBootstrapPaths, reference: dict[str, Any]
) -> MorpionPipelineGenerationManifest:
    """Attach a result at the sole commit without changing model, checkpoint or cursor."""
    from .checkpoint_patch_receipts import receipt_ledger

    with growth_receipt_ownership(paths, wait=True):
        state = load_bootstrap_run_state(paths.run_state_path)
        receipt_ledger(state)
        proposal = _proposal(paths, reference)
        admission = proposal["admission"]
        generation = admission["generation"]
        record = publication_record(state, generation)
        with generation_read_pin(paths.work_dir / record["tree_snapshot_path"]):
            _validate(paths, state, proposal, verify=True)
            previous = record.get(ACCEPTED)
            require(
                previous in (None, reference), "Conflicting accepted Training attempt."
            )
            if previous is None:
                if "parents" in admission:
                    require(
                        latest_candidate_parents(
                            paths, state, list(admission["parents"])
                        )
                        == admission["parents"],
                        "Stale Training parents: another accepted update advanced a candidate lineage.",
                    )
                    from .generation_publication_io import committed_evaluator

                    require(
                        committed_evaluator(paths.work_dir, state)[0]
                        == admission["incumbent"],
                        "Training incumbent changed before acceptance.",
                    )
                claim = load_pipeline_stage_claim(
                    paths.pipeline_training_claim_path_for_generation(generation)
                )
                require(
                    claim.claim_id == admission["claim_id"]
                    and claim.generation == generation
                    and claim.stage == "training",
                    "Stale Training claim was replaced.",
                )
                sync_artifact(
                    _directory(paths, admission["attempt_id"]), paths.work_dir
                )
                state = with_accepted_result(state, generation, ACCEPTED, reference)
                save_bootstrap_run_state(state, paths.run_state_path, durable=True)
            else:
                sync_artifact(paths.run_state_path, paths.work_dir)
            path = paths.pipeline_manifest_path_for_generation(generation)
            manifest = dataset.project_dataset(
                paths,
                state,
                generation,
                load_pipeline_manifest(path) if path.exists() else None,
            )
            manifest = project_training(paths, state, manifest)
            _repair(paths, manifest, proposal)
            return manifest


def run_training(
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    *,
    ttl_seconds: float,
    owner: str | None,
) -> MorpionPipelineGenerationManifest:
    """Run unchanged bounded trainer into private outputs, then publish and accept."""
    from datetime import UTC, datetime

    from chipiron.environments.morpion.learning import (
        morpion_supervised_rows_source_from_path,
    )

    from .bootstrap_memory import (
        log_after_cycle_gc,
        memory_diagnostics_config_from_args,
    )
    from .bootstrap_paths import MorpionBootstrapPaths
    from .control import load_bootstrap_control
    from .cycle_training import (
        restrict_evaluators_config,
        train_and_select_evaluators_streaming,
    )
    from .performance import StageMeasurement
    from .pipeline_artifacts import MorpionPipelineTrainingStatusArtifact
    from .profiling.memory_diagnostics import MemoryDiagnostics

    round_measurement = StageMeasurement(cuda_device=args.training_device)

    evaluators = restrict_evaluators_config(
        args.resolved_evaluators_config(), args.training_evaluator_names
    )
    control = load_bootstrap_control(paths.control_path)
    config = configuration(args, evaluators, control)
    with admit_training(
        paths, manifest, config, ttl_seconds=ttl_seconds, owner=owner
    ) as attempt:
        if isinstance(attempt, MorpionPipelineGenerationManifest):
            return attempt
        # Existing trainers put their caches beside rows. A private byte-for-byte copy
        # keeps those existing transient caches out of the immutable accepted Dataset.
        assert manifest.rows_path is not None
        rows = attempt.directory / "input" / "rows.jsonl"
        copy_artifact(inside(paths.work_dir, manifest.rows_path), rows)
        require(
            file_digest(rows) == attempt.admission["dataset"]["rows_sha256"],
            "Training input copy changed.",
        )
        private_paths = MorpionBootstrapPaths.from_work_dir(attempt.directory)
        state = load_bootstrap_run_state(paths.run_state_path)
        state = replace(
            state,
            latest_model_bundle_paths={
                name: str(paths.resolve_work_dir_path(parent["model"]["path"]))
                for name, parent in attempt.admission["parents"].items()
                if parent is not None
            }
            if "parents" in attempt.admission
            else {
                name: str(paths.resolve_work_dir_path(path))
                for name, path in (state.latest_model_bundle_paths or {}).items()
            },
            metadata={
                **state.metadata,
                "candidate_training_parents": attempt.admission["parents"],
            }
            if "parents" in attempt.admission
            else state.metadata,
        )
        memory = MemoryDiagnostics(memory_diagnostics_config_from_args(args))
        try:
            result = train_and_select_evaluators_streaming(
                args=args,
                paths=private_paths,
                run_state=state,
                rows_path=rows,
                rows_source=morpion_supervised_rows_source_from_path(rows),
                generation=manifest.generation,
                timestamp_utc=datetime.now(UTC).isoformat(),
                resolved_evaluators_config=evaluators,
                resolved_control=control,
                memory=memory,
                max_rows=args.training_max_rows,
                chunk_size=args.training_row_chunk_size,
            )
        finally:
            log_after_cycle_gc(memory)
            memory.close()
        require(
            file_digest(rows) == attempt.admission["dataset"]["rows_sha256"],
            "Training input changed during computation.",
        )
        selection = (
            _compare_incumbent(paths, private_paths, args, attempt, result, rows)
            if "parents" in attempt.admission
            else None
        )
        # The synchronous trainer has returned; only this attempt owned these
        # inputs/caches. Keep the accepted Dataset as the sole durable row source.
        shutil.rmtree(rows.parent)
        status = MorpionPipelineTrainingStatusArtifact(
            generation=manifest.generation,
            status="done",
            updated_at_utc=datetime.now(UTC).isoformat(),
            selected_evaluator_name=result.selected_evaluator_name,
            selection_policy=result.selection_policy,
            evaluator_results=result.evaluator_results,
            metadata={
                "training_evaluator_names": list(evaluators.evaluators),
                "training_duration_s": result.training_duration_s,
                "round_performance": round_measurement.finish(),
                **(
                    {
                        "candidate_execution": read_object(
                            attempt.directory / "candidate_execution" / "execution.json"
                        )
                    }
                    if (
                        attempt.directory / "candidate_execution" / "execution.json"
                    ).exists()
                    else {}
                ),
                **({"continual_selection": selection} if selection is not None else {}),
            },
        )
        return accept_training(paths, publish_training(paths, attempt, status))


def _compare_incumbent(
    paths: MorpionBootstrapPaths,
    private_paths: MorpionBootstrapPaths,
    args: MorpionBootstrapArgs,
    attempt: TrainingAttempt,
    result: Any,
    rows: Path,
) -> dict[str, Any]:
    """Compare all models on this accepted Dataset, never on another round's loss."""
    import math
    from dataclasses import fields

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

    populations = []
    metrics = {}
    for name, relative in result.model_bundle_paths.items():
        candidate_path = private_paths.resolve_work_dir_path(relative)
        require(candidate_path is not None, "Candidate output path is missing.")
        assert candidate_path is not None
        candidate_manifest = read_object(candidate_path / "morpion_manifest.json")
        populations.append(candidate_manifest["metadata"]["state_split"])
        loss = result.evaluator_metrics[name].validation_loss
        require(
            loss is not None and math.isfinite(loss),
            "Every candidate needs finite held-out loss.",
        )
        metrics[name] = loss
    require(
        bool(populations)
        and all(p == populations[0] for p in populations)
        and populations[0]["validation_count"] > 0,
        "Candidate validation populations differ or are empty.",
    )
    incumbent = attempt.admission["incumbent"]
    bundle = inside(paths.work_dir, incumbent["path"])
    require(artifact_digest(bundle) == incumbent["sha256"], "Incumbent model changed.")
    with torch.random.fork_rng(devices=[]):
        model, model_args, manifest = load_morpion_model_bundle(bundle)
    require(
        manifest.target_kind == "backup_value",
        "Incumbent target contract is incompatible.",
    )
    keywords = {
        f.name: getattr(model_args, f.name)
        for f in fields(MorpionTrainingArgs)
        if hasattr(model_args, f.name)
    }
    evaluation_args = MorpionTrainingArgs(
        **keywords,
        dataset_file=rows,
        output_dir=attempt.directory / "unused-evaluation",
        batch_size=64 if model_args.model_kind in {"linear", "mlp"} else 8,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        stable_validation_split=True,
        device=args.training_device,
    )
    evaluation_args, population = prepare_state_split(
        evaluation_args, args.training_max_rows
    )
    require(population == populations[0], "Incumbent validation population differs.")
    device = resolve_torch_device(args.training_device)
    stats = evaluate_streaming_metrics(
        model=model.to(device),
        args=evaluation_args,
        row_chunk_size=args.training_row_chunk_size,
        max_rows=args.training_max_rows,
        split="validation",
        device=device,
    )
    require(
        stats.sample_count == population["validation_count"]
        and math.isfinite(stats.loss),
        "Incumbent validation is empty or nonfinite.",
    )
    winner = min(metrics, key=lambda name: (metrics[name], name))
    require(
        winner == result.selected_evaluator_name,
        "Candidate selection is not deterministic.",
    )
    policy = args.continual_learning
    assert policy is not None
    round_number = attempt.admission["round_number"]
    eligible = (
        round_number % policy.selection_every_rounds == 0
        and round_number % policy.activation_every_rounds == 0
    )
    return {
        "round_number": round_number,
        "candidate": winner,
        "candidate_validation_losses": metrics,
        "validation_population": population,
        "incumbent": {"model": incumbent, "validation_loss": stats.loss},
        "improves_incumbent": metrics[winner] < stats.loss,
        "activation_eligible": eligible and metrics[winner] < stats.loss,
        "decision": "candidate"
        if eligible and metrics[winner] < stats.loss
        else "retain_incumbent",
        "tie_break": "incumbent_on_equal_else_candidate_name",
    }
