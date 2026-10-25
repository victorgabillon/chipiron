"""Accept immutable Training results without activating their models in Growth."""

from __future__ import annotations

import json
import os
import platform
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
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
from .pipeline_claims import claim_pipeline_stage, release_pipeline_stage_claim
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
    from pathlib import Path

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
                "selection": "forced_else_lowest_finite_validation_else_final_v1",
                "max_rows": args.training_max_rows,
                "chunk_size": args.training_row_chunk_size,
                "skip_diagnostics": args.skip_evaluator_diagnostics,
                "diagnostics_max_rows": args.evaluator_diagnostics_max_rows,
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
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        require(enabled(state), "Durable Training is not enabled.")
        record = publication_record(state, generation)
        if ACCEPTED in record:
            proposal = _proposal(paths, record[ACCEPTED])
            require(
                proposal["admission"]["configuration"] == config,
                "Conflicting accepted Training configuration.",
            )
            result = project_training(paths, state, manifest, verify=True)
            _repair(paths, result, proposal)
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
                    "schema_version": 1,
                    "attempt_id": identifier,
                    "generation": generation,
                    "identity": workspace_epoch(state),
                    "dataset": source,
                    "claim_id": claim.claim_id,
                    "configuration": config,
                    "reproducibility": reproducibility(),
                }
                atomic_json(staging / "admission.json", admission)
                attempt = TrainingAttempt(admission, staging)
                result = attempt
    try:
        yield result
    finally:
        if attempt is not None:
            try:
                with growth_receipt_ownership(paths):
                    if (
                        claim_path.is_file()
                        and load_pipeline_stage_claim(claim_path).claim_id
                        == attempt.admission["claim_id"]
                    ):
                        release_pipeline_stage_claim(
                            claim_path=claim_path,
                            claim_id=attempt.admission["claim_id"],
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
        and isinstance(admission["schema_version"], int)
        and not isinstance(admission["schema_version"], bool)
        and admission["schema_version"] == 1,
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
        )
        for name in names
        for r in [status.evaluator_results[name]]
    }
    require(
        select_or_force_active_evaluator_name(
            evaluator_metrics=metrics,
            force_evaluator=admission["configuration"]["force_evaluator"],
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
    if verify:
        require(
            dataset.accepted_dataset_identity(paths, state, generation)
            == admission["dataset"],
            "Training Dataset bytes changed.",
        )
    return status


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

    with growth_receipt_ownership(paths):
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
    from .pipeline_artifacts import MorpionPipelineTrainingStatusArtifact
    from .profiling.memory_diagnostics import MemoryDiagnostics

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
                name: str(paths.resolve_work_dir_path(path))
                for name, path in (state.latest_model_bundle_paths or {}).items()
            },
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
            },
        )
        return accept_training(paths, publish_training(paths, attempt, status))
