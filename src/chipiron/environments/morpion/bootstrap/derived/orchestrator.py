"""Bounded serial generation barriers with a durable restart journal."""

from __future__ import annotations

import gc
import random
import time
from pathlib import Path  # noqa: TC003 - public protocol introspection
from typing import Any, Protocol

import numpy as np
import torch

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineActiveModel,
    load_pipeline_active_model,
    load_pipeline_manifest,
    load_pipeline_stage_claim,
    load_pipeline_training_status_file,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    run_morpion_reevaluation_worker_once,
)

from .provenance import (
    DerivationError,
    atomic_json,
    inside,
    load_provenance,
    read_json,
    sha256,
)
from .runtime import args_from_config, drain_patch, make_runner

STAGES = ("growth", "dataset", "training", "reevaluation", "drain")


class GenerationStages(Protocol):
    """Each completed call must leave verifiable artifacts for the requested generation."""

    def run_stage(self, stage: str, generation: int) -> dict[str, Any]:
        """Run or recover exactly one stage, never choosing a newer generation."""
        ...


def run_sequential(
    work_dir: Path, stages: GenerationStages, *, max_wall_seconds: float = 12 * 3600
) -> dict[str, Any]:
    """No polling, independent workers or skipped generations; stop on any failure."""
    provenance = load_provenance(work_dir)
    journal_path = work_dir / "derived_orchestration.json"
    journal = read_json(journal_path)
    if journal.get("experiment_id") != provenance["experiment_id"]:
        message = "Orchestration identity does not match provenance."
        raise DerivationError(message)
    first = provenance["source_tree_generation"] + 1
    final = provenance["expected_final_generation"]
    if not first - 1 <= journal["completed_generation"] <= final:
        message = "Orchestration generation is outside its fixed bounds."
        raise DerivationError(message)
    started = time.monotonic()
    for generation in range(journal["completed_generation"] + 1, final + 1):
        entry = journal["generations"].setdefault(
            str(generation), {"completed_stages": [], "evidence": {}}
        )
        completed = entry["completed_stages"]
        if completed != list(STAGES[: len(completed)]):
            message = "Orchestration stage journal is not a sequential prefix."
            raise DerivationError(message)
        for stage in STAGES[len(completed) :]:
            if time.monotonic() - started >= max_wall_seconds:
                message = "Time budget reached at a recoverable stage boundary; resume explicitly."
                raise DerivationError(message)
            journal["phase"] = stage
            entry["inflight"] = stage
            entry["stage_started_unix_s"] = time.time()
            atomic_json(journal_path, journal)
            print(
                f"[derived] generation={generation}/{final} stage={stage} start",
                flush=True,
            )
            try:
                evidence = stages.run_stage(stage, generation)
            except BaseException as exc:
                entry["failure"] = {
                    "stage": stage,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                atomic_json(journal_path, journal)
                raise
            evidence["started_unix_s"] = entry["stage_started_unix_s"]
            evidence["finished_unix_s"] = time.time()
            entry["evidence"][stage] = evidence
            entry["completed_stages"].append(stage)
            entry.pop("inflight", None)
            entry.pop("failure", None)
            atomic_json(journal_path, journal)
            print(
                f"[derived] generation={generation}/{final} stage={stage} done",
                flush=True,
            )
        journal["completed_generation"] = generation
        journal["phase"] = "complete" if generation == final else "ready"
        atomic_json(journal_path, journal)
    return journal


class PipelineGenerationStages:
    """Reuse production stage implementations behind explicit derived barriers."""

    def __init__(self, work_dir: Path) -> None:
        """Load the immutable derived policy and persisted scientific config."""
        self.work_dir = work_dir
        self.provenance = load_provenance(work_dir)
        self.config = load_bootstrap_config(work_dir / "bootstrap_config.json")
        self.args = args_from_config(work_dir, self.config)
        self.paths = MorpionBootstrapPaths.from_work_dir(work_dir)
        self.claim_owner = f"derived:{self.provenance['experiment_id']}"

    def run_stage(self, stage: str, generation: int) -> dict[str, Any]:
        """Dispatch only explicitly named stages after checking immutable run policy."""
        if (
            sha256(self.paths.bootstrap_config_path)
            != self.provenance["prepared_config_sha256"]
        ):
            message = "Derived scientific config changed since preparation."
            raise DerivationError(message)
        if self.paths.control_path.exists():
            message = "Derived sequential execution forbids control overrides."
            raise DerivationError(message)
        if stage == "growth":
            return self._growth(generation)
        if stage == "dataset":
            return self._dataset(generation)
        if stage == "training":
            return self._training(generation)
        if stage == "reevaluation":
            return self._reevaluation(generation)
        if stage == "drain":
            return drain_patch(
                self.work_dir,
                self.config,
                self.provenance["rng_reset_policy"]["search"]["seed"],
            )
        message = f"Unknown derived stage: {stage}"
        raise DerivationError(message)

    def _growth(self, generation: int) -> dict[str, Any]:
        state = read_json(self.paths.run_state_path)
        if state["generation"] == generation - 1:
            if self.paths.pipeline_reevaluation_patch_path.exists():
                message = "Growth cannot cross a pending reevaluation barrier."
                raise DerivationError(message)
            runner = make_runner(
                self.work_dir,
                self.config,
                self.provenance["rng_reset_policy"]["search"]["seed"],
            )
            before = state["tree_size_at_last_save"]
            started = time.monotonic()
            result = run_pipeline_growth_stage(
                self.args, runner, max_cycles=1, derived_sequential=True
            )
            elapsed = time.monotonic() - started
            measured_steps = getattr(runner, "last_growth_steps", None)
            actual_steps = (
                measured_steps
                if isinstance(measured_steps, int)
                and not isinstance(measured_steps, bool)
                else None
            )
            del runner
            gc.collect()
            if result.generation != generation:
                message = "Growth stopped without publishing the required next generation (memory/budget/no growth)."
                raise DerivationError(message)
            state = read_json(self.paths.run_state_path)
            calibration: dict[str, Any] = {
                "growth_steps": actual_steps,
                "requested_growth_steps": self.args.max_growth_steps_per_cycle,
                "nodes_added": state["tree_size_at_last_save"] - before,
                "cycle_wall_s": elapsed,
            }
            calibration["nodes_per_step"] = (
                calibration["nodes_added"] / actual_steps if actual_steps else None
            )
        elif state["generation"] == generation:
            calibration = {"recovered_committed_growth": True}
        else:
            message = "Unexpected tree generation; refusing to skip or regrow it."
            raise DerivationError(message)
        checkpoint = inside(self.work_dir, state["latest_runtime_checkpoint_path"])
        export = read_json(inside(self.work_dir, state["latest_tree_snapshot_path"]))
        if (
            read_json(checkpoint / "manifest.json")["total_node_count"]
            != export["node_count"]
            or export["generation"] != generation
        ):
            message = "Committed Growth checkpoint/export do not agree."
            raise DerivationError(message)
        return {
            "checkpoint": state["latest_runtime_checkpoint_path"],
            "export": state["latest_tree_snapshot_path"],
            "nodes": export["node_count"],
            **calibration,
        }

    def _recover_claim(self, generation: int, stage: str) -> None:
        claim = (
            self.paths.pipeline_dataset_claim_path_for_generation(generation)
            if stage == "dataset"
            else self.paths.pipeline_training_claim_path_for_generation(generation)
        )
        if claim.exists():
            artifact = load_pipeline_stage_claim(claim)
            if artifact.owner != self.claim_owner:
                message = (
                    "Stage claim belongs to another operator; refusing to steal it."
                )
                raise DerivationError(message)
            claim.unlink()  # Only within the exclusive derived orchestration lock.

    def _dataset(self, generation: int) -> dict[str, Any]:
        self._recover_claim(generation, "dataset")
        result = run_pipeline_dataset_stage(
            self.args, generation=generation, claim_owner=self.claim_owner
        )
        if result.dataset_status != "done" or result.rows_path is None:
            message = "Dataset did not finish; generation remains incomplete."
            raise DerivationError(message)
        inside(self.work_dir, result.rows_path)
        return {
            "generation": generation,
            "rows_path": result.rows_path,
            "status": "done",
        }

    def _training(self, generation: int) -> dict[str, Any]:
        self._recover_claim(generation, "training")
        manifest_path = self.paths.pipeline_manifest_path_for_generation(generation)
        manifest = load_pipeline_manifest(manifest_path)
        status_path = self.paths.pipeline_training_status_path_for_generation(
            generation
        )
        status = (
            load_pipeline_training_status_file(status_path)
            if status_path.exists()
            else None
        )
        expected = set(self.config.evaluators.evaluators)
        complete = (
            manifest.training_status == "done"
            and status is not None
            and status.status == "done"
            and status.generation == generation
            and set(status.evaluator_results) == expected
        )
        if not complete:
            # Restart this generation from its original deterministic training initialization.
            cursor = read_json(self.paths.pipeline_training_cursor_path)
            if cursor.get("latest_completed_generation", generation - 1) >= generation:
                message = "Training cursor claims completion without complete evaluator evidence."
                raise DerivationError(message)
            cursor["latest_started_generation"] = generation - 1
            atomic_json(self.paths.pipeline_training_cursor_path, cursor)
            seed = self.provenance["rng_reset_policy"]["training"]["seed"] + generation
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            manifest = run_pipeline_training_stage(
                self.args, generation=generation, claim_owner=self.claim_owner
            )
            status = load_pipeline_training_status_file(status_path)
        if (
            status is None
            or manifest.training_status != "done"
            or status.status != "done"
            or status.generation != generation
            or set(status.evaluator_results) != expected
            or manifest.selected_evaluator_name != status.selected_evaluator_name
        ):
            message = "Training is incomplete; no selection from a partial evaluator family is allowed."
            raise DerivationError(message)
        active = load_pipeline_active_model(self.paths.pipeline_active_model_path)
        if active.generation != generation:
            # Recover a crash between complete training-status persistence and active publication.
            from chipiron.environments.morpion.bootstrap.pipeline.cursors import (
                active_model_generation_for_training_guard,
            )

            bound = active_model_generation_for_training_guard(self.paths)
            if bound is not None and bound >= generation:
                message = "Local model publication guard rejected this generation."
                raise DerivationError(message)
            selected = status.selected_evaluator_name
            if selected is None:
                message = "Completed training lacks a selected evaluator."
                raise DerivationError(message)
            save_pipeline_active_model(
                MorpionPipelineActiveModel(
                    generation=generation,
                    evaluator_name=selected,
                    model_bundle_path=manifest.model_bundle_paths[selected],
                    updated_at_utc=status.updated_at_utc,
                    metadata={
                        "selection_policy": status.selection_policy,
                        "derived_experiment_id": self.provenance["experiment_id"],
                    },
                ),
                self.paths.pipeline_active_model_path,
            )
        elif (
            active.evaluator_name != status.selected_evaluator_name
            or active.model_bundle_path
            != manifest.model_bundle_paths[active.evaluator_name]
        ):
            message = "Published evaluator differs from training selection."
            raise DerivationError(message)
        atomic_json(
            self.paths.pipeline_training_cursor_path,
            {
                "latest_started_generation": generation,
                "latest_completed_generation": generation,
            },
        )
        return {
            "generation": generation,
            "evaluators": sorted(expected),
            "selected": status.selected_evaluator_name,
            "selection_policy": status.selection_policy,
        }

    def _reevaluation(self, generation: int) -> dict[str, Any]:
        state = read_json(self.paths.run_state_path)
        receipt = state["metadata"].get("derived_drain", {})
        if receipt.get("generation") == generation and receipt.get("full_pass"):
            return {"recovered_committed_drain": True}
        if not self.paths.pipeline_reevaluation_patch_path.exists():
            result = run_morpion_reevaluation_worker_once(
                self.args,
                max_nodes_per_patch=state["tree_size_at_last_save"],
                patch_id=f"derived-{self.provenance['experiment_id'][:16]}-{generation}",
            )
            if not result.patch_written:
                message = (
                    f"Reevaluation did not produce the required patch: {result.reason}"
                )
                raise DerivationError(message)
        patch = read_json(self.paths.pipeline_reevaluation_patch_path)
        if (
            patch["tree_generation"] != generation
            or not patch["metadata"]["completed_full_pass"]
        ):
            message = "Reevaluation did not finish the complete generation."
            raise DerivationError(message)
        return {
            "generation": generation,
            "patch_id": patch["patch_id"],
            "full_pass": True,
        }
