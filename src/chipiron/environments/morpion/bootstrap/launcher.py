"""Canonical human/operator launcher for one persistent Morpion bootstrap run."""

from __future__ import annotations

import argparse
import logging
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from .anemone_runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    apply_runtime_control_to_runner_args,
)
from .bootstrap_args import MorpionBootstrapArgs
from .bootstrap_loop import (
    MorpionBootstrapPaths,
    run_morpion_bootstrap_loop,
)
from .config import (
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MorpionBootstrapConfig,
    bootstrap_config_from_args,
    load_bootstrap_config,
)
from .control import (
    MorpionBootstrapControl,
    effective_runtime_config_from_config_and_control,
    load_bootstrap_control,
)
from .evaluator_family import CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
from .history import MorpionBootstrapLatestStatus, load_latest_bootstrap_status
from .pipeline_config import (
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
    MorpionPipelineStage,
)
from .pipeline_orchestrator import (
    MorpionPipelineOrchestratorResult,
    run_morpion_artifact_pipeline_once,
)
from .pipeline_stages import (
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)
from .process_control import (
    mark_current_launcher_process_stopped,
    register_current_launcher_process,
)
from .reevaluation_worker import (
    MorpionReevaluationWorkerResult,
    run_morpion_reevaluation_worker_once,
)
from .run_state import MorpionBootstrapRunState, load_bootstrap_run_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .pipeline_artifacts import MorpionPipelineGenerationManifest
    from .pv_family_targets import PvFamilyTargetPolicy

LOGGER = logging.getLogger(__name__)


def _non_loop_stage_requires_artifact_pipeline_error() -> ValueError:
    """Build the canonical launcher mode mismatch error."""
    return ValueError("artifact_pipeline mode required for non-loop stages")


@dataclass(frozen=True, slots=True)
class MorpionBootstrapLauncherArgs:
    """Launcher-only options for the canonical Morpion operator entrypoint."""

    bootstrap_args: MorpionBootstrapArgs
    max_cycles: int | None = None
    pipeline_stage: MorpionPipelineStage = "loop"
    pipeline_generation: int | None = None
    reevaluation_max_nodes_per_patch: int = 10_000
    open_dashboard: bool = False
    print_startup_summary: bool = True
    print_dashboard_hint: bool = True

    @property
    def work_dir(self) -> Path:
        """Return the resolved launcher work directory."""
        return MorpionBootstrapPaths.from_work_dir(
            self.bootstrap_args.work_dir
        ).work_dir


@dataclass(frozen=True, slots=True)
class _LauncherStartupStatus:
    """Resolved startup information used by the canonical launcher."""

    paths: MorpionBootstrapPaths
    run_mode: str
    resolved_bootstrap_args: MorpionBootstrapArgs
    bootstrap_config: MorpionBootstrapConfig
    control: MorpionBootstrapControl
    run_state: MorpionBootstrapRunState | None
    latest_status: MorpionBootstrapLatestStatus | None
    resolved_evaluator_family_preset: str | None
    evaluator_family_source: Literal[
        "explicit",
        "launcher_default",
        "explicit_config",
        "legacy_default",
    ]
    resolved_evaluator_names: tuple[str, ...]
    config_exists: bool
    control_exists: bool
    run_state_exists: bool
    history_exists: bool
    latest_status_exists: bool


def run_morpion_bootstrap_experiment(
    launcher_args: MorpionBootstrapLauncherArgs,
) -> (
    MorpionBootstrapRunState
    | MorpionPipelineGenerationManifest
    | MorpionPipelineOrchestratorResult
    | MorpionReevaluationWorkerResult
):
    """Run one persistent Morpion bootstrap experiment end to end.

    This launcher is the canonical human/operator entrypoint for one real
    Morpion bootstrap experiment backed by the Anemone search runner.
    """
    LOGGER.info("[launcher] startup_start work_dir=%s", str(launcher_args.work_dir))
    startup_status = _collect_launcher_startup_status(launcher_args)
    LOGGER.info(
        "[launcher] startup_done mode=%s evaluators=%s max_cycles=%s",
        startup_status.run_mode,
        len(startup_status.resolved_evaluator_names),
        "none" if launcher_args.max_cycles is None else str(launcher_args.max_cycles),
    )
    if launcher_args.print_startup_summary:
        print(
            _render_launcher_startup_summary(
                startup_status,
                dashboard_requested=launcher_args.open_dashboard,
            )
        )
    if launcher_args.print_dashboard_hint:
        if launcher_args.print_startup_summary:
            print()
        print(
            _render_dashboard_hint(
                startup_status.paths.work_dir,
                requested_open=launcher_args.open_dashboard,
            )
        )

    if startup_status.resolved_bootstrap_args.pipeline_mode == "single_process":
        if launcher_args.pipeline_stage != "loop":
            raise _non_loop_stage_requires_artifact_pipeline_error()
        runner = _build_launcher_runner(startup_status)
        LOGGER.info("[launcher] runner_ready")
        return run_morpion_bootstrap_loop(
            startup_status.resolved_bootstrap_args,
            runner,
            max_cycles=launcher_args.max_cycles,
        )

    if launcher_args.pipeline_stage == "loop":
        runner = _build_launcher_runner(startup_status)
        LOGGER.info("[launcher] runner_ready")
        return run_morpion_artifact_pipeline_once(
            startup_status.resolved_bootstrap_args,
            runner,
            max_growth_cycles=1
            if launcher_args.max_cycles is None
            else launcher_args.max_cycles,
        )
    if launcher_args.pipeline_stage == "reevaluation":
        return run_morpion_reevaluation_worker_once(
            startup_status.resolved_bootstrap_args,
            max_nodes_per_patch=launcher_args.reevaluation_max_nodes_per_patch,
        )
    if launcher_args.pipeline_stage == "dataset":
        assert launcher_args.pipeline_generation is not None
        return run_pipeline_dataset_stage(
            startup_status.resolved_bootstrap_args,
            generation=launcher_args.pipeline_generation,
        )
    if launcher_args.pipeline_stage == "training":
        assert launcher_args.pipeline_generation is not None
        return run_pipeline_training_stage(
            startup_status.resolved_bootstrap_args,
            generation=launcher_args.pipeline_generation,
        )

    runner = _build_launcher_runner(startup_status)
    LOGGER.info("[launcher] runner_ready")
    return run_pipeline_growth_stage(
        startup_status.resolved_bootstrap_args,
        runner,
        max_cycles=1 if launcher_args.max_cycles is None else launcher_args.max_cycles,
    )


def _resolve_launcher_bootstrap_args(
    launcher_args: MorpionBootstrapLauncherArgs,
) -> MorpionBootstrapArgs:
    """Apply launcher-only default evaluator selection to bootstrap args."""
    bootstrap_args = launcher_args.bootstrap_args
    if bootstrap_args.evaluators_config is not None:
        return bootstrap_args
    if bootstrap_args.evaluator_family_preset is not None:
        return bootstrap_args
    return replace(
        bootstrap_args,
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )


def _launcher_evaluator_family_source(
    bootstrap_args: MorpionBootstrapArgs,
    resolved_bootstrap_args: MorpionBootstrapArgs,
) -> Literal[
    "explicit",
    "launcher_default",
    "explicit_config",
    "legacy_default",
]:
    """Classify how the launcher resolved the effective evaluator selection path."""
    if bootstrap_args.evaluators_config is not None:
        return "explicit_config"
    if bootstrap_args.evaluator_family_preset is not None:
        return "explicit"
    if resolved_bootstrap_args.evaluator_family_preset is not None:
        return "launcher_default"
    return "legacy_default"


def _collect_launcher_startup_status(
    launcher_args: MorpionBootstrapLauncherArgs,
) -> _LauncherStartupStatus:
    """Resolve the operator-facing bootstrap status before entering the loop."""
    resolved_bootstrap_args = _resolve_launcher_bootstrap_args(launcher_args)
    resolved_evaluator_family_source = _launcher_evaluator_family_source(
        launcher_args.bootstrap_args,
        resolved_bootstrap_args,
    )
    paths = MorpionBootstrapPaths.from_work_dir(resolved_bootstrap_args.work_dir)
    config_exists = paths.bootstrap_config_path.is_file()
    control_exists = paths.control_path.is_file()
    run_state_exists = paths.run_state_path.is_file()
    history_exists = paths.history_jsonl_path.is_file()
    latest_status_exists = paths.latest_status_path.is_file()

    bootstrap_config = bootstrap_config_from_args(resolved_bootstrap_args)
    if config_exists:
        bootstrap_config = load_bootstrap_config(paths.bootstrap_config_path)

    control = load_bootstrap_control(paths.control_path)
    run_state = (
        load_bootstrap_run_state(paths.run_state_path) if run_state_exists else None
    )
    latest_status = (
        load_latest_bootstrap_status(paths.latest_status_path)
        if latest_status_exists
        else None
    )

    return _LauncherStartupStatus(
        paths=paths,
        run_mode=_resolve_run_mode(
            config_exists=config_exists,
            control_exists=control_exists,
            run_state_exists=run_state_exists,
            history_exists=history_exists,
            latest_status_exists=latest_status_exists,
        ),
        resolved_bootstrap_args=resolved_bootstrap_args,
        bootstrap_config=bootstrap_config,
        control=control,
        run_state=run_state,
        latest_status=latest_status,
        resolved_evaluator_family_preset=resolved_bootstrap_args.evaluator_family_preset,
        evaluator_family_source=resolved_evaluator_family_source,
        resolved_evaluator_names=tuple(sorted(bootstrap_config.evaluators.evaluators)),
        config_exists=config_exists,
        control_exists=control_exists,
        run_state_exists=run_state_exists,
        history_exists=history_exists,
        latest_status_exists=latest_status_exists,
    )


def _build_launcher_runner(
    startup_status: _LauncherStartupStatus,
) -> AnemoneMorpionSearchRunner:
    """Construct the canonical real Anemone runner for the experiment."""
    effective_runtime_config = effective_runtime_config_from_config_and_control(
        startup_status.bootstrap_config,
        startup_status.control,
    )
    runner_args = apply_runtime_control_to_runner_args(
        AnemoneMorpionSearchRunnerArgs(),
        effective_runtime_config,
    )
    return AnemoneMorpionSearchRunner(runner_args)


def _render_launcher_startup_summary(
    startup_status: _LauncherStartupStatus,
    *,
    dashboard_requested: bool,
) -> str:
    """Render the operator-facing launcher summary without printing."""
    effective_runtime_config = effective_runtime_config_from_config_and_control(
        startup_status.bootstrap_config,
        startup_status.control,
    )
    baseline_tree_branch_limit = (
        startup_status.bootstrap_config.runtime.tree_branch_limit
    )
    control_tree_branch_limit = startup_status.control.runtime.tree_branch_limit
    evaluators = ", ".join(startup_status.resolved_evaluator_names)
    control_fragment = (
        "none" if control_tree_branch_limit is None else str(control_tree_branch_limit)
    )
    return "\n".join(
        (
            "=== Morpion Bootstrap Launcher ===",
            f"work dir: {startup_status.paths.work_dir}",
            f"mode: {startup_status.run_mode}",
            f"bootstrap config: {_render_config_state(startup_status.config_exists)}",
            f"control file: {_render_presence(startup_status.control_exists)}",
            f"run state: {_render_presence(startup_status.run_state_exists)}",
            f"history: {_render_presence(startup_status.history_exists)}",
            f"latest status: {_render_presence(startup_status.latest_status_exists)}",
            f"latest generation: {_render_optional_int(_latest_generation(startup_status))}",
            f"latest cycle: {_render_optional_int(_latest_cycle_index(startup_status))}",
            f"evaluator family preset: {_render_evaluator_family_line(startup_status)}",
            f"configured evaluators: {evaluators}",
            f"forced evaluator control: {_render_optional_text(startup_status.control.force_evaluator)}",
            "tree_branch_limit: "
            f"{effective_runtime_config.tree_branch_limit} "
            f"(baseline {baseline_tree_branch_limit}, control override {control_fragment})",
            "dashboard: "
            f"{'requested via separate process hint' if dashboard_requested else 'available via separate process'}",
            "paths:",
            f"  config: {startup_status.paths.bootstrap_config_path}",
            f"  control: {startup_status.paths.control_path}",
            f"  run state: {startup_status.paths.run_state_path}",
            f"  history: {startup_status.paths.history_jsonl_path}",
            f"  latest status: {startup_status.paths.latest_status_path}",
            f"  runtime checkpoints: {startup_status.paths.runtime_checkpoint_dir}",
            f"  tree snapshots: {startup_status.paths.tree_snapshot_dir}",
            f"  rows: {startup_status.paths.rows_dir}",
            f"  models: {startup_status.paths.model_dir}",
        )
    )


def _render_dashboard_hint(work_dir: Path, *, requested_open: bool) -> str:
    """Render the exact dashboard command for the current work directory."""
    command = (
        "python -m chipiron.environments.morpion.bootstrap.dashboard_app "
        f"--work-dir {shlex.quote(str(work_dir))}"
    )
    heading = (
        "Dashboard requested: start it in a separate terminal for this work dir."
        if requested_open
        else "Dashboard available for this work dir."
    )
    return "\n".join((heading, f"command: {command}"))


def build_launcher_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the canonical bootstrap launcher."""
    parser = argparse.ArgumentParser(
        description=(
            "Canonical human/operator launcher for one persistent Morpion "
            "bootstrap experiment. By default, launcher-driven runs use the "
            "canonical 8-model Morpion evaluator family."
        )
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluator-family",
        type=str,
        default=None,
        help=(
            "Evaluator-family preset to use. If omitted, the launcher defaults "
            "to the canonical 8-model Morpion family unless explicit "
            "evaluators_config is supplied programmatically."
        ),
    )
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument(
        "--open-dashboard",
        "--dashboard",
        dest="open_dashboard",
        action="store_true",
    )
    parser.add_argument(
        "--no-print-startup-summary",
        dest="print_startup_summary",
        action="store_false",
    )
    parser.add_argument(
        "--no-print-dashboard-hint",
        dest="print_dashboard_hint",
        action="store_false",
    )
    parser.set_defaults(
        print_startup_summary=True,
        print_dashboard_hint=True,
    )
    parser.add_argument("--max-growth-steps-per-cycle", type=int, default=1000)
    parser.add_argument("--save-after-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--save-after-tree-growth-factor",
        type=float,
        default=2.0,
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--use-backed-up-value",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dataset-family-target-policy",
        choices=(
            "none",
            "pv_mean_prediction",
            "pv_min_prediction",
            "pv_blend_mean_prediction",
            "pv_blend_min_prediction",
            "pv_exact_then_mean_prediction",
            "pv_exact_then_min_prediction",
            "pv_exact_then_blend_mean_prediction",
            "pv_exact_then_blend_min_prediction",
        ),
        default="none",
    )
    parser.add_argument(
        "--dataset-family-prediction-blend",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--evaluator-update-policy",
        choices=["future_only", "reevaluate_all", "reevaluate_frontier"],
        default=DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
        help=(
            "How restored trees should use a newly selected evaluator. "
            "'future_only' keeps existing node values and uses the evaluator only for future expansions. "
            "'reevaluate_all' reevaluates existing nodes when supported. "
            "'reevaluate_frontier' is reserved for future partial reevaluation."
        ),
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=["single_process", "artifact_pipeline"],
        default=DEFAULT_MORPION_PIPELINE_MODE,
        help=(
            "Bootstrap execution mode. 'single_process' is the current in-process loop. "
            "'artifact_pipeline' enables the Phase 3 file-driven stage entrypoints."
        ),
    )
    parser.add_argument(
        "--pipeline-stage",
        choices=["loop", "growth", "dataset", "training", "reevaluation"],
        default="loop",
        help=(
            "Pipeline dispatch target. 'loop' preserves the current launcher behavior. "
            "Artifact-pipeline mode also supports 'growth', 'dataset', 'training', "
            "and 'reevaluation'."
        ),
    )
    parser.add_argument(
        "--pipeline-generation",
        type=int,
        default=None,
        help="Generation index required by the dataset and training pipeline stages.",
    )
    parser.add_argument(
        "--reevaluation-max-nodes-per-patch",
        type=int,
        default=10_000,
        help="Maximum nodes to include in one reevaluation patch.",
    )
    parser.add_argument(
        "--memory-diagnostics",
        action="store_true",
        help="Log lightweight process memory diagnostics at bootstrap cycle phases.",
    )
    parser.add_argument(
        "--memory-diagnostics-gc-growth",
        action="store_true",
        help="Also log aggregate Python GC object counts by type.",
    )
    parser.add_argument(
        "--memory-diagnostics-tracemalloc",
        action="store_true",
        help="Also log tracemalloc allocation diffs between memory checkpoints.",
    )
    parser.add_argument(
        "--memory-diagnostics-torch-tensors",
        action="store_true",
        help="Also log aggregate live torch.Tensor counts and storage bytes.",
    )
    parser.add_argument(
        "--memory-diagnostics-referrers",
        action="store_true",
        help="Also log bounded referrer chains for selected live GC object types.",
    )
    parser.add_argument(
        "--memory-diagnostics-referrer-type-pattern",
        action="append",
        default=[],
        help=(
            "Fully qualified type name, substring, or glob for referrer diagnostics. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--memory-diagnostics-referrer-max-objects-per-type",
        type=int,
        default=2,
        help="Number of matching objects to inspect per type for referrer diagnostics.",
    )
    parser.add_argument(
        "--memory-diagnostics-referrer-max-depth",
        type=int,
        default=2,
        help="Maximum recursive referrer depth to log.",
    )
    parser.add_argument(
        "--memory-diagnostics-top-n",
        type=int,
        default=20,
        help="Number of GC/tracemalloc entries to log per memory checkpoint.",
    )
    parser.add_argument(
        "--tree-branch-limit",
        type=int,
        default=DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    )
    return parser


def _validate_pipeline_stage_cli(
    *,
    parser: argparse.ArgumentParser,
    pipeline_mode: str,
    pipeline_stage: str,
    pipeline_generation: int | None,
) -> None:
    """Validate CLI stage selection against the chosen pipeline mode."""
    if pipeline_mode == "single_process" and pipeline_stage != "loop":
        parser.error(
            "--pipeline-stage is only valid with 'loop' when --pipeline-mode is single_process."
        )
    if pipeline_generation is not None and pipeline_stage not in {"dataset", "training"}:
        parser.error(
            "--pipeline-generation is only valid with --pipeline-stage dataset or training."
        )
    if (
        pipeline_mode == "artifact_pipeline"
        and pipeline_stage in {"dataset", "training"}
        and pipeline_generation is None
    ):
        parser.error(
            "--pipeline-generation is required for --pipeline-stage dataset and training."
        )


def launcher_args_from_cli(
    argv: Sequence[str] | None = None,
) -> MorpionBootstrapLauncherArgs:
    """Parse CLI arguments into the canonical launcher dataclass."""
    parser = build_launcher_argument_parser()
    parsed = parser.parse_args(argv)
    _validate_pipeline_stage_cli(
        parser=parser,
        pipeline_mode=parsed.pipeline_mode,
        pipeline_stage=parsed.pipeline_stage,
        pipeline_generation=parsed.pipeline_generation,
    )
    bootstrap_args = MorpionBootstrapArgs(
        work_dir=parsed.work_dir,
        evaluator_family_preset=parsed.evaluator_family,
        max_growth_steps_per_cycle=parsed.max_growth_steps_per_cycle,
        save_after_seconds=parsed.save_after_seconds,
        save_after_tree_growth_factor=parsed.save_after_tree_growth_factor,
        max_rows=parsed.max_rows,
        use_backed_up_value=parsed.use_backed_up_value,
        evaluator_update_policy=parsed.evaluator_update_policy,
        pipeline_mode=parsed.pipeline_mode,
        dataset_family_target_policy=cast(
            "PvFamilyTargetPolicy",
            parsed.dataset_family_target_policy,
        ),
        dataset_family_prediction_blend=parsed.dataset_family_prediction_blend,
        memory_diagnostics=parsed.memory_diagnostics,
        memory_diagnostics_gc_growth=parsed.memory_diagnostics_gc_growth,
        memory_diagnostics_tracemalloc=parsed.memory_diagnostics_tracemalloc,
        memory_diagnostics_torch_tensors=parsed.memory_diagnostics_torch_tensors,
        memory_diagnostics_referrers=parsed.memory_diagnostics_referrers,
        memory_diagnostics_referrer_type_patterns=tuple(
            parsed.memory_diagnostics_referrer_type_pattern
        ),
        memory_diagnostics_referrer_max_objects_per_type=(
            parsed.memory_diagnostics_referrer_max_objects_per_type
        ),
        memory_diagnostics_referrer_max_depth=(
            parsed.memory_diagnostics_referrer_max_depth
        ),
        memory_diagnostics_top_n=parsed.memory_diagnostics_top_n,
        tree_branch_limit=parsed.tree_branch_limit,
    )
    return MorpionBootstrapLauncherArgs(
        bootstrap_args=bootstrap_args,
        max_cycles=parsed.max_cycles,
        pipeline_stage=cast("MorpionPipelineStage", parsed.pipeline_stage),
        pipeline_generation=parsed.pipeline_generation,
        reevaluation_max_nodes_per_patch=parsed.reevaluation_max_nodes_per_patch,
        open_dashboard=parsed.open_dashboard,
        print_startup_summary=parsed.print_startup_summary,
        print_dashboard_hint=parsed.print_dashboard_hint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical Morpion bootstrap launcher CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    launcher_args = launcher_args_from_cli(argv)
    paths = MorpionBootstrapPaths.from_work_dir(launcher_args.bootstrap_args.work_dir)
    register_current_launcher_process(paths)
    exit_code = 0
    try:
        run_morpion_bootstrap_experiment(launcher_args)
    except BaseException:
        exit_code = 1
        raise
    else:
        return exit_code
    finally:
        mark_current_launcher_process_stopped(paths, exit_code=exit_code)


def _resolve_run_mode(
    *,
    config_exists: bool,
    control_exists: bool,
    run_state_exists: bool,
    history_exists: bool,
    latest_status_exists: bool,
) -> str:
    """Classify the work directory as fresh, resume, or partial."""
    if run_state_exists:
        return "resume"
    if any((config_exists, control_exists, history_exists, latest_status_exists)):
        return "partial artifacts"
    return "fresh run"


def _latest_generation(startup_status: _LauncherStartupStatus) -> int | None:
    """Return the latest known generation for summary rendering."""
    if startup_status.run_state is not None:
        return startup_status.run_state.generation
    if startup_status.latest_status is not None:
        return startup_status.latest_status.latest_generation
    return None


def _latest_cycle_index(startup_status: _LauncherStartupStatus) -> int | None:
    """Return the latest known cycle index for summary rendering."""
    if startup_status.run_state is not None:
        return startup_status.run_state.cycle_index
    if startup_status.latest_status is not None:
        return startup_status.latest_status.latest_cycle_index
    return None


def _render_config_state(config_exists: bool) -> str:
    """Render whether the persisted config already exists."""
    if config_exists:
        return "present"
    return "absent; will be written from launcher args"


def _render_presence(is_present: bool) -> str:
    """Render a stable presence/absence marker."""
    return "present" if is_present else "absent"


def _render_optional_text(value: str | None) -> str:
    """Render optional text consistently in launcher output."""
    return "none" if value is None else value


def _render_evaluator_family_line(startup_status: _LauncherStartupStatus) -> str:
    """Render one explicit evaluator-family provenance line for operators."""
    source = startup_status.evaluator_family_source
    preset = startup_status.resolved_evaluator_family_preset
    if source == "explicit_config":
        return "none (explicit evaluators_config)"
    if source == "explicit" and preset is not None:
        return f"{preset} (explicit)"
    if source == "launcher_default" and preset is not None:
        return f"{preset} (launcher default)"
    if source == "legacy_default":
        return "none (legacy default)"
    return _render_optional_text(preset)


def _render_optional_int(value: int | None) -> str:
    """Render optional integers consistently in launcher output."""
    return "n/a" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MorpionBootstrapLauncherArgs",
    "build_launcher_argument_parser",
    "launcher_args_from_cli",
    "main",
    "run_morpion_bootstrap_experiment",
]
