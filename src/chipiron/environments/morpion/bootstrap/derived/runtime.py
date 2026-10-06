"""Derived-only strict restoration and durable reevaluation drain, with no Growth."""

from __future__ import annotations

import gc
import uuid
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import (
    MorpionBootstrapConfig,  # noqa: TC001
)
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    effective_runtime_config_from_config_and_control,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    delete_reevaluation_patch,
    load_pipeline_active_model,
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    log_available_ram_guard,
)
from chipiron.environments.morpion.bootstrap.reevaluation_patch_consumer import (
    apply_pending_reevaluation_patch_to_runner,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)

from .provenance import DerivationError, atomic_json, inside, read_json, sync_checkpoint


def args_from_config(
    work_dir: Path, config: MorpionBootstrapConfig
) -> MorpionBootstrapArgs:
    """Forward every persisted scientific field; defaults cover diagnostics only."""
    runtime = asdict(config.runtime)
    dataset = asdict(config.dataset)
    dataset["dataset_family_prediction_blend"] = dataset.pop("family_prediction_blend")
    dataset["dataset_family_target_policy"] = dataset.pop("family_target_policy")
    return MorpionBootstrapArgs(
        work_dir=work_dir,
        **runtime,
        **dataset,
        search=config.search,
        evaluators_config=config.evaluators,
        validation_fraction=config.validation_fraction,
        validation_seed=config.validation_seed,
        training_device=config.training_device,
        evaluator_update_policy=config.evaluator_update_policy,
        pipeline_mode=config.pipeline_mode,
        training_export_mode=config.training_export_mode,
    )


class DerivedSearchRunner(AnemoneMorpionSearchRunner):
    """Fail closed rather than creating a root or reading a training export."""

    def __init__(self, work_dir: Path, args: AnemoneMorpionSearchRunnerArgs) -> None:
        """Bind strict checkpoint guards to one derived workspace."""
        super().__init__(args)
        self.work_dir = work_dir

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Restore the committed runtime checkpoint; reject every fallback path."""
        state = load_bootstrap_run_state(self.work_dir / "run_state.json")
        if state.latest_runtime_checkpoint_path is None or tree_snapshot_path is None:
            message = "Derived execution requires a runtime checkpoint; fresh initialization is forbidden."
            raise DerivationError(message)
        expected = inside(self.work_dir, state.latest_runtime_checkpoint_path)
        if (
            Path(tree_snapshot_path).resolve() != expected.resolve()
            or not expected.is_dir()
        ):
            message = (
                "Derived restore refuses fallback checkpoints or training exports."
            )
            raise DerivationError(message)
        manifest = read_json(expected / "manifest.json")
        if manifest["total_node_count"] != state.tree_size_at_last_save:
            message = "Derived checkpoint node count disagrees with run_state."
            raise DerivationError(message)
        active = load_pipeline_active_model(
            self.work_dir / "pipeline/active_model.json"
        )
        bundle = inside(self.work_dir, active.model_bundle_path)
        if (
            model_bundle_path is None
            or Path(model_bundle_path).resolve() != bundle.resolve()
        ):
            message = "Derived restore requires the persisted active model."
            raise DerivationError(message)
        inside(bundle, "param.pt")
        super().load_or_create(
            expected, bundle, effective_runtime_config, reevaluate_tree=reevaluate_tree
        )
        if self.current_tree_size() != state.tree_size_at_last_save:
            message = "Runtime restore produced an unexpected tree size."
            raise DerivationError(message)


def make_runner(
    work_dir: Path, config: MorpionBootstrapConfig, search_seed: int
) -> DerivedSearchRunner:
    """Use the normal runner with strict input guards; RNG is restored from checkpoint."""
    runtime = asdict(config.runtime)
    accepted = {item.name for item in fields(AnemoneMorpionSearchRunnerArgs)}
    return DerivedSearchRunner(
        work_dir,
        AnemoneMorpionSearchRunnerArgs(
            search_args=default_search_args(rollout=config.search.rollout),
            random_seed=search_seed,
            **{key: value for key, value in runtime.items() if key in accepted},
        ),
    )


def drain_patch(
    work_dir: Path, config: MorpionBootstrapConfig, search_seed: int
) -> dict[str, Any]:
    """Commit a patched runtime checkpoint before acknowledgement; never call grow."""
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    state = load_bootstrap_run_state(paths.run_state_path)
    committed = state.metadata.get("derived_drain", {})
    if not paths.pipeline_reevaluation_patch_path.exists():
        if committed.get("generation") == state.generation and committed.get(
            "full_pass"
        ):
            return cast("dict[str, Any]", committed)
        message = "No pending patch and no durable drain receipt."
        raise DerivationError(message)
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    if committed.get("patch_id") == patch.patch_id:
        delete_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
        return cast("dict[str, Any]", committed)
    active = load_pipeline_active_model(paths.pipeline_active_model_path)
    if (
        patch.tree_generation != state.generation
        or patch.evaluator_generation != active.generation
        or patch.model_bundle_path != active.model_bundle_path
        or len(patch.rows) != state.tree_size_at_last_save
        or not patch.metadata.get("completed_full_pass")
    ):
        message = "Derived drain requires a complete matching generation/model patch."
        raise DerivationError(message)
    if not log_available_ram_guard(
        stage="growth",
        generation=state.generation,
        action="derived_drain_restore",
        required_mb=config.runtime.min_available_ram_mb,
    ):
        message = "Memory guard stopped the derived drain; patch remains pending."
        raise DerivationError(message)
    runner = make_runner(work_dir, config, search_seed)
    runtime_config = effective_runtime_config_from_config_and_control(
        config, MorpionBootstrapControl()
    )
    if state.latest_runtime_checkpoint_path is None:
        message = "Missing committed runtime checkpoint."
        raise DerivationError(message)
    runner.load_or_create(
        inside(work_dir, state.latest_runtime_checkpoint_path),
        inside(work_dir, active.model_bundle_path),
        runtime_config,
        reevaluate_tree=False,
    )
    result = apply_pending_reevaluation_patch_to_runner(
        paths=paths, runner=runner, delete_after_apply=False
    )
    if not result.patch_applied or result.num_rows != len(patch.rows):
        message = "Derived drain did not apply all required rows."
        raise DerivationError(message)
    # A unique uncommitted checkpoint is harmless after interruption; run_state is the commit point.
    output = (
        work_dir
        / "derived_checkpoints"
        / f"generation_{state.generation:06d}"
        / f"drain_{uuid.uuid4().hex}.sharded"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    runner.save_checkpoint(output)
    sync_checkpoint(output)
    receipt = {
        "generation": state.generation,
        "patch_id": patch.patch_id,
        "rows": result.num_rows,
        "full_pass": True,
        "growth_steps": 0,
        "checkpoint": str(output.relative_to(work_dir)),
    }
    next_state = asdict(state)
    next_state["active_evaluator_name"] = active.evaluator_name
    next_state["latest_runtime_checkpoint_path"] = receipt["checkpoint"]
    next_state["metadata"]["runtime_checkpoint_path"] = receipt["checkpoint"]
    next_state["metadata"]["derived_drain"] = receipt
    atomic_json(paths.run_state_path, next_state)
    delete_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    del runner
    gc.collect()
    return receipt
