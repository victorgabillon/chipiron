"""One real worker per process for the explicit research-only E2E recipe."""

from __future__ import annotations

import argparse
import json
import logging
import random
import resource
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionPipelineActiveModel,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.config import (
    MorpionBootstrapRolloutConfig,
    MorpionBootstrapSearchConfig,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    save_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    committed_evaluator,
    file_digest,
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    MorpionBootstrapLauncherArgs,
    run_morpion_bootstrap_experiment,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    run_morpion_reevaluation_worker_once,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from chipiron.environments.morpion.bootstrap.worker_result_io import workspace_epoch
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
    save_morpion_model_bundle,
)
from docs.research.atomic_activation.measure import disk
from scripts.benchmark_morpion_scientific_reader import costs, delta


def arguments(root: Path) -> MorpionBootstrapArgs:
    """Reconstruct the same explicit frozen configuration in each fresh process."""
    return MorpionBootstrapArgs(
        work_dir=root,
        pipeline_mode="artifact_pipeline",
        training_export_mode="sharded",
        runtime_checkpoint_format="sharded",
        growth_skip_training_export=False,
        max_growth_steps_per_cycle=1,
        save_after_seconds=0.0,
        tree_branch_limit=100000,
        reevaluation_blend_alpha=1.0,
        dataset_confidence_policy="selected_value_v2",
        dataset_family_target_policy="none",
        require_exact_or_terminal=False,
        use_backed_up_value=False,
        min_visit_count=0,
        max_rows=16,
        validation_seed=17,
        validation_fraction=0.25,
        training_device="cpu",
        training_row_chunk_size=16,
        skip_evaluator_diagnostics=True,
        search=MorpionBootstrapSearchConfig(
            rollout=MorpionBootstrapRolloutConfig(
                enabled=True, max_extra_steps=2, random_seed=23
            )
        ),
        evaluators_config=MorpionEvaluatorsConfig(
            evaluators={
                name: MorpionEvaluatorSpec(
                    name=name,
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=4,
                    learning_rate=rate,
                )
                for name, rate in (("linear_a", 0.001), ("linear_b", 0.003))
            }
        ),
    )


def initialize(root: Path, count: int, endgame: bool) -> None:
    """Genuine legal search, real seed weights, independently validated fresh genesis."""
    from contextlib import nullcontext

    args = arguments(root)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    assert not paths.run_state_path.exists()
    paths.ensure_directories()
    config = bootstrap_config_from_args(args)
    atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
    # Exercise the existing operator control file, without changing scientific policy.
    save_bootstrap_control(
        MorpionBootstrapControl(max_growth_steps_per_cycle=1, save_after_seconds=0.0),
        paths.control_path,
    )
    model_args = MorpionRegressorArgs(model_kind="linear")
    model = build_morpion_regressor(model_args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        list(model.parameters())[-1].fill_(42)
    bundle = root / "seed"
    save_morpion_model_bundle(model, bundle, model_args=model_args)
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name="seed",
            model_bundle_path="seed",
            updated_at_utc="2026-10-24T00:00:00Z",
            source="external_seed",
            source_generation=1,
        ),
        paths.pipeline_active_model_path,
    )
    context = nullcontext()
    if endgame:
        from tests.environments.test_morpion_value_certainty import endgame as fixture

        context = patch(
            "chipiron.environments.morpion.bootstrap.runtime.runner.initial_state",
            lambda: fixture.__wrapped__()[0],
        )
    with context:
        runner = AnemoneMorpionSearchRunner(
            AnemoneMorpionSearchRunnerArgs(
                random_seed=4,
                search_args=default_search_args(rollout=args.search.rollout),
                runtime_checkpoint_format="sharded",
            )
        )
        runner.load_or_create(
            None, bundle, MorpionBootstrapEffectiveRuntimeConfig(args.tree_branch_limit)
        )
    while runner.current_tree_size() < (2 if endgame else count):
        runner.grow(1)
    checkpoint = root / "genesis-checkpoint"
    runner.save_checkpoint(checkpoint)
    producer = {
        "revision": "4ada24712dca95d1114d3e838a13c3447af998ac",
        "objective": "maximum_total_moves",
        "evaluator": artifact_digest(bundle),
        "config_sha256": bootstrap_config_sha256(config),
    }
    source = runner.export_closed_scientific_snapshot(
        root / "genesis-science", generation=1, producer=producer
    )
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        last_save_unix_s=1e9,
        tree_size_at_last_save=runner.current_tree_size(),
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
        latest_tree_snapshot_path=paths.relative_to_work_dir(source),
        metadata={
            "bootstrap_config_hash": bootstrap_config_sha256(config),
            "bootstrap_effective_runtime": {
                "tree_branch_limit": args.tree_branch_limit,
                "reevaluation_blend_alpha": 1.0,
            },
        },
    )
    runner.discard_uncommitted_runtime()
    initialize_scientific_v2_publication(
        paths,
        state,
        producer=producer,
        closed_scientific_snapshot=True,
        atomic_reevaluation=True,
        growth_scoring=True,
        fenced_dataset=True,
        durable_training=True,
        model_activation=True,
    )


def authority(paths: MorpionBootstrapPaths) -> dict:
    """Resolve verified disk authority; never infer adoption from a status filename."""
    state = load_bootstrap_run_state(paths.run_state_path)
    head = publication_records(state.metadata)[-1]
    validate_capsule(paths.work_dir, head)
    model, _ = committed_evaluator(paths.work_dir, state)
    return {
        "run_state_sha256": file_digest(paths.run_state_path),
        "generation": state.generation,
        "nodes": state.tree_size_at_last_save,
        "head": head,
        "model": model,
        "workspace_epoch": workspace_epoch(state),
    }


def main() -> None:
    """No scheduler: execute exactly one requested stage and terminate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("root", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--generation", type=int)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--endgame", action="store_true")
    parser.add_argument("--request", type=Path)
    options = parser.parse_args()
    root = options.root.resolve()
    paths = MorpionBootstrapPaths.from_work_dir(root)
    evidence = root.parent / "evidence"
    evidence.mkdir(exist_ok=True, parents=True)
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101 + (options.generation or 0))
    random.seed(19)
    np.random.seed(19)
    args = arguments(root)
    before = costs()
    measurement = StageMeasurement()
    prior = authority(paths) if paths.run_state_path.exists() else None
    extra = {}
    # Redirect only optional leaderboard side effects into this new experiment.
    original = stages.persist_certified_leaderboard_candidates
    with patch.object(
        stages,
        "persist_certified_leaderboard_candidates",
        lambda **kw: original(**kw, leaderboard_path=root.parent / "leaderboard.jsonl"),
    ):
        if options.stage == "init":
            initialize(root, options.count, options.endgame)
        elif options.stage == "growth":
            state = load_bootstrap_run_state(paths.run_state_path)
            runner, _ = _runner(paths, state)
            producer = publication_records(state.metadata)[-1]["scientific_identity"][
                "producer"
            ]["identity"]
            stages.run_pipeline_growth_stage(
                args,
                runner,
                max_cycles=1,
                checkpoint_bound_receipts=True,
                durable_generation_publication=True,
                scientific_v2_producer=producer,
                closed_scientific_snapshot=True,
            )
        elif options.stage == "reevaluation":
            result = run_morpion_reevaluation_worker_once(
                args,
                max_nodes_per_patch=7,
                use_streaming_sharded_reader=True,
                allow_scientific_v2=True,
            )
            assert result.patch_written
            extra["worker_result"] = asdict(result)
        elif options.stage == "dataset":
            stages.run_pipeline_dataset_stage(
                args,
                generation=options.generation,
                use_streaming_sharded_reader=True,
                allow_scientific_v2=True,
            )
        elif options.stage == "training":
            run_morpion_bootstrap_experiment(
                MorpionBootstrapLauncherArgs(
                    bootstrap_args=args,
                    pipeline_stage="training",
                    pipeline_generation=options.generation,
                    print_startup_summary=False,
                    print_dashboard_hint=False,
                    quiet_worker_startup=True,
                )
            )
        elif options.stage == "prepare":
            request = prepare_model_activation_request(
                paths, training_generation=options.generation
            )
            assert options.request is not None
            options.request.write_text(json.dumps(request, indent=2) + "\n")
        elif options.stage == "activate":
            assert options.request is not None
            activate_model(paths, json.loads(options.request.read_text()))
        elif options.stage == "predict-growth":
            from .checks import growth_oracle

            growth_oracle(root, evidence / options.label)
            extra["oracle"] = str(evidence / options.label / "expected.json")
        elif options.stage in {"check", "restart"}:
            from .checks import verify

            extra = verify(
                root, evidence / options.label, full=options.stage == "check"
            )
        else:
            raise ValueError(options.stage)
    observed = authority(paths)
    result = {
        "stage": options.stage,
        "label": options.label,
        "before": prior,
        "after": observed,
        **delta(before),
        **measurement.finish(),
        "disk": disk(root),
        **extra,
    }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["process_cpu_user_s"] = usage.ru_utime
    result["process_cpu_system_s"] = usage.ru_stime
    result["process_peak_rss_mib"] = usage.ru_maxrss / 1024
    (evidence / f"{options.label}.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
