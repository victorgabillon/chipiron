"""Retained frozen search to fresh authority using the existing genesis publisher."""

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .generation_publication_io import (
    artifact_digest,
    copy_artifact,
    object_digest,
    require,
)

if TYPE_CHECKING:
    from .bootstrap_args import MorpionBootstrapArgs
    from .config import MorpionBootstrapConfig
    from .run_state import MorpionBootstrapRunState


def initialize_fork_genesis(
    args: MorpionBootstrapArgs,
    config: MorpionBootstrapConfig,
    *,
    checkpoint: Path,
    incumbent: Path,
    incumbent_name: str,
) -> MorpionBootstrapRunState:
    """Caller owns verified copied inputs; final-root capability is already constructed.

    Only the physical private root is used here. Normal managed execution must
    reject it until publication at the config's intended canonical root.
    """
    from importlib import import_module
    from platform import python_version

    import anemone
    import torch

    import chipiron
    from chipiron.learning.torch_runtime import torch_execution_profile_metadata

    from .bootstrap_paths import MorpionBootstrapPaths
    from .config import bootstrap_config_sha256, save_bootstrap_config
    from .control import MorpionBootstrapEffectiveRuntimeConfig
    from .derived.cli import package_fingerprint
    from .execution_profile import policy_metadata, require_args_profile
    from .fork_origin import KEY
    from .learning_schedule import context_marker, initial_window
    from .pipeline_artifacts import (
        MorpionPipelineActiveModel,
        save_pipeline_active_model,
    )
    from .run_state import initialize_bootstrap_run_state
    from .runtime.runner import (
        AnemoneMorpionSearchRunner,
        AnemoneMorpionSearchRunnerArgs,
        default_search_args,
    )
    from .scientific_publication_initialization import (
        initialize_scientific_v2_publication,
    )
    from .scientific_stream import ENCODING

    require_args_profile(args, stage="initialize")
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    require(
        not paths.run_state_path.exists(),
        "Fork genesis requires fresh private authority.",
    )
    paths.ensure_directories()
    genesis = paths.work_dir / "genesis"
    genesis.mkdir(exist_ok=True)
    model = genesis / "incumbent"
    copy_artifact(incumbent, model)
    save_bootstrap_config(config, paths.bootstrap_config_path)
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name=incumbent_name,
            model_bundle_path=paths.relative_to_work_dir(model),
            updated_at_utc="1970-01-01T00:00:00Z",
            source="external_seed",
            source_generation=1,
        ),
        paths.pipeline_active_model_path,
    )
    producer = {
        "revision": package_fingerprint(Path(chipiron.__file__).parent),
        "anemone_source_sha256": package_fingerprint(Path(anemone.__file__).parent),
        "objective": "maximum_total_moves",
        "evaluator": artifact_digest(model),
        "config_sha256": bootstrap_config_sha256(config),
        "python_version": python_version(),
        "torch_version": str(torch.__version__),
        "torch_execution_profile_sha256": object_digest(
            torch_execution_profile_metadata()
        ),
    }
    for name in ("coral", "atomheart", "valanga", "parsley"):
        module = import_module(name)
        require(module.__file__ is not None, f"Missing {name} source.")
        assert module.__file__ is not None
        producer[name + "_source_sha256"] = package_fingerprint(
            Path(module.__file__).parent
        )
    effective = MorpionBootstrapEffectiveRuntimeConfig(
        args.tree_branch_limit, reevaluation_blend_alpha=args.reevaluation_blend_alpha
    )
    runner = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            runtime_checkpoint_format="sharded",
            search_args=default_search_args(rollout=config.search.rollout),
        )
    )
    try:
        runner.load_fork_genesis(
            checkpoint,
            model,
            effective,
            checkpoint_sha256=artifact_digest(checkpoint),
            model_sha256=artifact_digest(model),
        )
        runner.validate_model_refresh_certificates()
        output = genesis / "checkpoint"
        runner.save_checkpoint(output)
        snapshot = runner.export_closed_scientific_snapshot(
            genesis / "science", generation=1, producer=producer, encoding=ENCODING
        )
        metadata: dict[str, Any] = {
            **policy_metadata(args),
            **torch_execution_profile_metadata(),
            "continual_learning": context_marker(args.learning_schedule),
            **initial_window(args.learning_schedule),
            "bootstrap_config_hash": producer["config_sha256"],
            "bootstrap_effective_runtime": {
                "tree_branch_limit": args.tree_branch_limit,
                "reevaluation_blend_alpha": args.reevaluation_blend_alpha,
            },
            **({KEY: config.metadata[KEY]} if KEY in config.metadata else {}),
        }
        state = replace(
            initialize_bootstrap_run_state(),
            generation=1,
            tree_size_at_last_save=runner.current_tree_size(),
            last_save_unix_s=time.time(),
            latest_runtime_checkpoint_path=paths.relative_to_work_dir(output),
            latest_tree_snapshot_path=paths.relative_to_work_dir(snapshot),
            active_evaluator_name=incumbent_name,
            metadata=metadata,
        )
    finally:
        runner.discard_uncommitted_runtime()
    return initialize_scientific_v2_publication(
        paths,
        state,
        producer=producer,
        closed_scientific_snapshot=True,
        atomic_reevaluation=True,
        fenced_dataset=True,
        durable_training=True,
        growth_scoring=True,
        model_activation=True,
    )
