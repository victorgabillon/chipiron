"""Canonical human/operator launcher for one persistent Morpion bootstrap run."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from anemone.checkpoints import DEFAULT_CHECKPOINT_FILE_FORMAT, checkpoint_cli_name

from .anemone_runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    _default_search_args,
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
    MorpionBootstrapRolloutConfig,
    MorpionBootstrapSearchConfig,
    bootstrap_config_from_args,
    load_bootstrap_config,
    save_bootstrap_config,
    validate_stage_bootstrap_config_compatibility,
)
from .control import (
    MorpionBootstrapControl,
    effective_runtime_config_from_config_and_control,
    load_bootstrap_control,
)
from .evaluator_family import (
    CANONICAL_LINEAR_MLP_GRAPH_SMALL_MORPION_EVALUATOR_FAMILY_PRESET,
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
)
from .history import MorpionBootstrapLatestStatus, load_latest_bootstrap_status
from .pipeline_config import (
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
    DEFAULT_MORPION_TRAINING_EXPORT_MODE,
    MorpionPipelineStage,
)
from .pipeline_orchestrator import (
    MorpionPipelineOrchestratorResult,
    MorpionPipelineWorkerResult,
    run_morpion_artifact_pipeline_once,
    run_next_pipeline_dataset_stage_once,
    run_next_pipeline_training_stage_once,
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


type CheckpointLoggerSetter = Callable[[int], None]

LOGGER = logging.getLogger(__name__)


_ROLLOUT_CONFIG_CLI_PREFIXES = (
    "--rollout-max-extra-steps",
    "--rollout-action-selector-kind",
    "--rollout-random-seed",
)


def _non_loop_stage_requires_artifact_pipeline_error() -> ValueError:
    """Build the canonical launcher mode mismatch error."""
    return ValueError("artifact_pipeline mode required for non-loop stages")


def _rollout_max_extra_steps_parse_error() -> argparse.ArgumentTypeError:
    """Return the stable rollout max-extra-steps parse error."""
    return argparse.ArgumentTypeError("expected 'none' or a non-negative integer")


def _parse_optional_non_negative_int(raw: str) -> int | None:
    """Parse a non-negative integer or an explicit unbounded sentinel."""
    if raw.lower() in {"none", "null", "unbounded"}:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise _rollout_max_extra_steps_parse_error() from exc
    if value < 0:
        raise _rollout_max_extra_steps_parse_error()
    return value


def _parse_training_evaluator_names(raw: str | None) -> tuple[str, ...] | None:
    """Parse an optional comma-separated evaluator-name subset."""
    if raw is None:
        return None
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    return names or None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapLauncherArgs:
    """Launcher-only options for the canonical Morpion operator entrypoint."""

    bootstrap_args: MorpionBootstrapArgs
    max_cycles: int | None = None
    pipeline_stage: MorpionPipelineStage = "loop"
    pipeline_generation: int | None = None
    reevaluation_max_nodes_per_patch: int = 10_000
    verbose_checkpoint_logs: bool = False
    training_export_mode_explicit: bool = False
    rollout_config_explicit: bool = False
    min_available_ram_mb_explicit: bool = False
    tree_branch_limit_explicit: bool = False
    candidate_checkpoint_load_headroom_explicit: bool = False
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
    | MorpionPipelineWorkerResult
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
    if launcher_args.pipeline_stage == "dataset_worker":
        return run_next_pipeline_dataset_stage_once(
            startup_status.resolved_bootstrap_args
        )
    if launcher_args.pipeline_stage == "training_worker":
        return run_next_pipeline_training_stage_once(
            startup_status.resolved_bootstrap_args
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
    requested_bootstrap_args = resolved_bootstrap_args
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

    if config_exists:
        bootstrap_config = load_bootstrap_config(paths.bootstrap_config_path)
        if not launcher_args.training_export_mode_explicit:
            requested_bootstrap_args = replace(
                requested_bootstrap_args,
                training_export_mode=bootstrap_config.training_export_mode,
            )
        if not launcher_args.rollout_config_explicit:
            requested_bootstrap_args = replace(
                requested_bootstrap_args,
                search=bootstrap_config.search,
            )
        if not launcher_args.min_available_ram_mb_explicit:
            requested_bootstrap_args = replace(
                requested_bootstrap_args,
                min_available_ram_mb=bootstrap_config.runtime.min_available_ram_mb,
            )
        if not launcher_args.tree_branch_limit_explicit:
            requested_bootstrap_args = replace(
                requested_bootstrap_args,
                tree_branch_limit=bootstrap_config.runtime.tree_branch_limit,
            )
        if not launcher_args.candidate_checkpoint_load_headroom_explicit:
            requested_bootstrap_args = replace(
                requested_bootstrap_args,
                candidate_checkpoint_load_headroom_factor=(
                    bootstrap_config.runtime.candidate_checkpoint_load_headroom_factor
                ),
                candidate_checkpoint_load_min_headroom_mb=(
                    bootstrap_config.runtime.candidate_checkpoint_load_min_headroom_mb
                ),
            )
        requested_config = bootstrap_config_from_args(requested_bootstrap_args)
        bootstrap_config = _adopt_growth_rollout_config_if_requested(
            persisted_config=bootstrap_config,
            requested_config=requested_config,
            stage=launcher_args.pipeline_stage,
            rollout_config_explicit=launcher_args.rollout_config_explicit,
            config_path=paths.bootstrap_config_path,
        )
        validate_stage_bootstrap_config_compatibility(
            stage=launcher_args.pipeline_stage,
            persisted_config=bootstrap_config,
            requested_config=requested_config,
        )
        resolved_bootstrap_args = _bootstrap_args_with_persisted_config(
            requested_bootstrap_args,
            persisted_config=bootstrap_config,
        )
    else:
        bootstrap_config = bootstrap_config_from_args(requested_bootstrap_args)
        save_bootstrap_config(bootstrap_config, paths.bootstrap_config_path)

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


def _stage_can_adopt_growth_rollout_config(stage: MorpionPipelineStage) -> bool:
    """Return whether one launcher stage performs growth expansion work."""
    return stage in {"growth", "loop"}


def _adopt_growth_rollout_config_if_requested(
    *,
    persisted_config: MorpionBootstrapConfig,
    requested_config: MorpionBootstrapConfig,
    stage: MorpionPipelineStage,
    rollout_config_explicit: bool,
    config_path: Path,
) -> MorpionBootstrapConfig:
    """Persist requested rollout config changes for growth-capable stages."""
    if not rollout_config_explicit:
        return persisted_config
    if not _stage_can_adopt_growth_rollout_config(stage):
        return persisted_config
    if persisted_config.search.rollout == requested_config.search.rollout:
        return persisted_config

    adopted_config = replace(persisted_config, search=requested_config.search)
    LOGGER.info(
        "[config] adopting growth rollout config changes: %s",
        "; ".join(
            _rollout_config_change_fragments(
                previous=persisted_config,
                current=adopted_config,
            )
        ),
    )
    save_bootstrap_config(adopted_config, config_path)
    return adopted_config


def _rollout_config_change_fragments(
    *,
    previous: MorpionBootstrapConfig,
    current: MorpionBootstrapConfig,
) -> tuple[str, ...]:
    """Return stable log fragments for changed rollout config fields."""
    previous_rollout = previous.search.rollout
    current_rollout = current.search.rollout
    fields = (
        ("enabled", "search.rollout.enabled"),
        ("max_extra_steps", "search.rollout.max_extra_steps"),
        ("action_selector_kind", "search.rollout.action_selector_kind"),
        ("random_seed", "search.rollout.random_seed"),
        ("stop_on_existing_node", "search.rollout.stop_on_existing_node"),
    )
    return tuple(
        f"{config_field}: {getattr(previous_rollout, attr)!r} -> {getattr(current_rollout, attr)!r}"
        for attr, config_field in fields
        if getattr(previous_rollout, attr) != getattr(current_rollout, attr)
    )


def _bootstrap_args_with_persisted_config(
    args: MorpionBootstrapArgs,
    *,
    persisted_config: MorpionBootstrapConfig,
) -> MorpionBootstrapArgs:
    """Apply the persisted bootstrap config to one stage's runtime args."""
    return replace(
        args,
        save_after_tree_growth_factor=persisted_config.runtime.save_after_tree_growth_factor,
        save_after_seconds=persisted_config.runtime.save_after_seconds,
        require_exact_or_terminal=persisted_config.dataset.require_exact_or_terminal,
        min_depth=persisted_config.dataset.min_depth,
        min_visit_count=persisted_config.dataset.min_visit_count,
        max_rows=persisted_config.dataset.max_rows,
        use_backed_up_value=persisted_config.dataset.use_backed_up_value,
        dataset_family_target_policy=persisted_config.dataset.family_target_policy,
        dataset_family_prediction_blend=persisted_config.dataset.family_prediction_blend,
        evaluator_update_policy=persisted_config.evaluator_update_policy,
        pipeline_mode=persisted_config.pipeline_mode,
        training_export_mode=persisted_config.training_export_mode,
        search=persisted_config.search,
        evaluators_config=persisted_config.evaluators,
        evaluator_family_preset=None,
    )


def _build_launcher_runner(
    startup_status: _LauncherStartupStatus,
) -> AnemoneMorpionSearchRunner:
    """Construct the canonical real Anemone runner for the experiment."""
    effective_runtime_config = effective_runtime_config_from_config_and_control(
        startup_status.bootstrap_config,
        startup_status.control,
    )
    if startup_status.control.runtime.tree_branch_limit is None:
        effective_runtime_config = replace(
            effective_runtime_config,
            tree_branch_limit=startup_status.resolved_bootstrap_args.tree_branch_limit,
        )
    runner_args = apply_runtime_control_to_runner_args(
        AnemoneMorpionSearchRunnerArgs(
            search_args=_default_search_args(
                rollout=startup_status.bootstrap_config.search.rollout
            )
        ),
        effective_runtime_config,
    )
    return AnemoneMorpionSearchRunner(runner_args)


def _configure_anemone_checkpoint_logging(*, verbose_checkpoint_logs: bool) -> None:
    """Set checkpoint-internals verbosity for normal versus debug launcher runs."""
    checkpoint_logger_level_setter = _load_checkpoint_logger_level_setter()
    if checkpoint_logger_level_setter is None:
        LOGGER.warning(
            "[launcher] checkpoint_log_config_skipped reason=missing_anemone_setter"
        )
        return
    checkpoint_logger_level_setter(
        logging.DEBUG if verbose_checkpoint_logs else logging.INFO
    )


def _load_checkpoint_logger_level_setter() -> CheckpointLoggerSetter | None:
    """Resolve the optional Anemone checkpoint logger setter lazily."""
    try:
        from anemone.utils.logger import set_checkpoint_logger_level
    except ImportError:
        return None
    return cast("CheckpointLoggerSetter", set_checkpoint_logger_level)


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
    resolved_tree_branch_limit = (
        startup_status.resolved_bootstrap_args.tree_branch_limit
        if control_tree_branch_limit is None
        else effective_runtime_config.tree_branch_limit
    )
    latest_runtime_checkpoint_path = _latest_runtime_checkpoint_path(startup_status)
    latest_training_artifact_path = _latest_training_artifact_path(startup_status)
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
            (
                "training export mode: "
                f"{startup_status.bootstrap_config.training_export_mode} "
                f"({_render_training_export_mode_note(startup_status.bootstrap_config.training_export_mode)})"
            ),
            (
                "runtime checkpoint format: "
                f"{checkpoint_cli_name(DEFAULT_CHECKPOINT_FILE_FORMAT)} "
                "(default; legacy .json checkpoints still load)"
            ),
            f"latest runtime checkpoint: {_render_optional_text(latest_runtime_checkpoint_path)}",
            f"latest training artifact: {_render_optional_text(latest_training_artifact_path)}",
            f"evaluator family preset: {_render_evaluator_family_line(startup_status)}",
            f"configured evaluators: {evaluators}",
            f"forced evaluator control: {_render_optional_text(startup_status.control.force_evaluator)}",
            "tree_branch_limit: "
            f"{resolved_tree_branch_limit} "
            f"(baseline {baseline_tree_branch_limit}, control override {control_fragment})",
            "rollout: "
            f"enabled={startup_status.bootstrap_config.search.rollout.enabled} "
            f"max_extra_steps={startup_status.bootstrap_config.search.rollout.max_extra_steps} "
            f"action_selector_kind={startup_status.bootstrap_config.search.rollout.action_selector_kind} "
            f"random_seed={startup_status.bootstrap_config.search.rollout.random_seed} "
            f"stop_on_existing_node={startup_status.bootstrap_config.search.rollout.stop_on_existing_node}",
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
            f"  sharded tree snapshots: {startup_status.paths.sharded_tree_snapshot_dir}",
            f"  rows: {startup_status.paths.rows_dir}",
            f"  models: {startup_status.paths.model_dir}",
        )
    )


def _render_training_export_mode_note(training_export_mode: str) -> str:
    """Return one short operator-facing note for the current export mode."""
    if training_export_mode == "sharded":
        return "default"
    if training_export_mode == "both":
        return "compatibility/debug"
    return "legacy compatibility/debug"


def _latest_runtime_checkpoint_path(
    startup_status: _LauncherStartupStatus,
) -> str | None:
    """Return the newest known runtime checkpoint path from persisted state."""
    latest_event = (
        startup_status.latest_status.latest_event
        if startup_status.latest_status
        else None
    )
    latest_event_path = (
        None if latest_event is None else latest_event.artifacts.runtime_checkpoint_path
    )
    if latest_event_path is not None:
        return latest_event_path
    run_state = startup_status.run_state
    return None if run_state is None else run_state.latest_runtime_checkpoint_path


def _latest_training_artifact_path(
    startup_status: _LauncherStartupStatus,
) -> str | None:
    """Return the newest known flat export or sharded generation manifest path."""
    latest_event = (
        startup_status.latest_status.latest_event
        if startup_status.latest_status
        else None
    )
    latest_event_path = (
        None if latest_event is None else latest_event.artifacts.tree_snapshot_path
    )
    if latest_event_path is not None:
        return latest_event_path
    run_state = startup_status.run_state
    return None if run_state is None else run_state.latest_tree_snapshot_path


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
            "evaluators_config is supplied programmatically. Available presets "
            f"include {CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET!r} and "
            f"{CANONICAL_LINEAR_MLP_GRAPH_SMALL_MORPION_EVALUATOR_FAMILY_PRESET!r}."
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
    parser.add_argument(
        "--require-exact-or-terminal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--min-depth", type=int, default=None)
    parser.add_argument("--min-visit-count", type=int, default=None)
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
        "--training-export-mode",
        choices=["flat", "sharded", "both"],
        default=DEFAULT_MORPION_TRAINING_EXPORT_MODE,
        help=(
            "Training export artifact format. 'sharded' is the normal/default mode. "
            "'flat' preserves the legacy single-file compatibility export, and 'both' "
            "writes sharded artifacts alongside a flat compatibility/debug export."
        ),
    )
    parser.add_argument(
        "--pipeline-stage",
        choices=[
            "loop",
            "growth",
            "dataset",
            "dataset_worker",
            "training",
            "training_worker",
            "reevaluation",
        ],
        default="loop",
        help=(
            "Pipeline dispatch target. 'loop' preserves the current launcher behavior. "
            "Artifact-pipeline mode also supports 'growth', 'dataset', "
            "'dataset_worker', 'training', 'training_worker', and 'reevaluation'."
        ),
    )
    parser.add_argument(
        "--pipeline-generation",
        type=int,
        default=None,
        help="Generation index required by the dataset and training pipeline stages.",
    )
    parser.add_argument(
        "--training-evaluator-names",
        type=str,
        default=None,
        help="Comma-separated evaluator names to train in the pipeline training stage.",
    )
    parser.add_argument(
        "--training-max-rows",
        type=_parse_optional_non_negative_int,
        default=None,
        help="Maximum rows to use during pipeline training; default uses all rows.",
    )
    parser.add_argument(
        "--training-row-chunk-size",
        type=int,
        default=8192,
        help="Rows per chunk for streaming JSONL pipeline training.",
    )
    parser.add_argument(
        "--skip-evaluator-diagnostics",
        action="store_true",
        help="Skip evaluator diagnostics during pipeline training.",
    )
    parser.add_argument(
        "--evaluator-diagnostics-max-rows",
        type=_parse_optional_non_negative_int,
        default=60,
        help=(
            "Maximum rows used for evaluator diagnostics. Use 'none' to allow "
            "full diagnostics where supported."
        ),
    )
    parser.add_argument(
        "--growth-memory-profile",
        action="store_true",
        help="Log opt-in shallow memory attribution for growth runtimes.",
    )
    parser.add_argument(
        "--growth-memory-profile-top-n",
        type=int,
        default=20,
        help="Number of GC type and runner attribute entries in growth profiles.",
    )
    parser.add_argument(
        "--growth-memory-profile-sample-nodes",
        type=int,
        default=2000,
        help="Maximum live tree nodes to sample for growth memory profiles.",
    )
    parser.add_argument(
        "--candidate-checkpoint-load-headroom-factor",
        type=float,
        default=60.0,
        help=(
            "Multiplier applied to compressed candidate-checkpoint size when "
            "forecasting pre-load RAM headroom."
        ),
    )
    parser.add_argument(
        "--candidate-checkpoint-load-min-headroom-mb",
        type=int,
        default=512,
        help="Minimum estimated RAM headroom for candidate checkpoint load.",
    )
    parser.add_argument(
        "--reevaluation-max-nodes-per-patch",
        type=int,
        default=10_000,
        help="Maximum nodes to include in one reevaluation patch.",
    )
    parser.add_argument(
        "--verbose-checkpoint-logs",
        action="store_true",
        help=(
            "Enable detailed Anemone checkpoint restore/build internals such as "
            "restore phases and delta-candidate rejection logs."
        ),
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
        "--min-available-ram-mb",
        type=_parse_optional_non_negative_int,
        default=None,
        help=(
            "Minimum available RAM in MiB required before heavy artifact-pipeline "
            "loads; none or 0 disables the guard."
        ),
    )
    parser.add_argument(
        "--tree-branch-limit",
        type=int,
        default=DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    )
    parser.add_argument(
        "--rollout-after-opening",
        action="store_true",
        default=False,
        help="Enable Anemone rollout expansion after each selected opening.",
    )
    parser.add_argument(
        "--rollout-max-extra-steps",
        type=_parse_optional_non_negative_int,
        default=None,
        help="'none' for unbounded-until-stop rollout, or a non-negative integer.",
    )
    parser.add_argument(
        "--rollout-action-selector-kind",
        choices=[
            "first_openable",
            "random_openable",
            "no_rollout",
            "first_legal_prefer_openable",
            "random_legal_prefer_openable",
        ],
        default="random_legal_prefer_openable",
    )
    parser.add_argument("--rollout-random-seed", type=int, default=0)
    parser.add_argument(
        "--rollout-stop-on-existing-node",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--reevaluation-blend-alpha",
        type=float,
        default=1.0,
        help=(
            "Blend new reevaluation direct values with existing live values. "
            "1.0 replaces values exactly; lower values smooth updates."
        ),
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
    if pipeline_generation is not None and pipeline_stage not in {
        "dataset",
        "training",
    }:
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
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    training_export_mode_explicit = False
    rollout_config_explicit = False
    training_export_mode_explicit = any(
        argument == "--training-export-mode"
        or argument.startswith("--training-export-mode=")
        for argument in argv_list
    )
    min_available_ram_mb_explicit = any(
        argument == "--min-available-ram-mb"
        or argument.startswith("--min-available-ram-mb=")
        for argument in argv_list
    )
    tree_branch_limit_explicit = any(
        argument == "--tree-branch-limit"
        or argument.startswith("--tree-branch-limit=")
        for argument in argv_list
    )
    candidate_checkpoint_load_headroom_explicit = any(
        argument == "--candidate-checkpoint-load-headroom-factor"
        or argument == "--candidate-checkpoint-load-min-headroom-mb"
        or argument.startswith(
            (
                "--candidate-checkpoint-load-headroom-factor=",
                "--candidate-checkpoint-load-min-headroom-mb=",
            )
        )
        for argument in argv_list
    )
    rollout_config_explicit = any(
        argument == "--rollout-after-opening"
        or argument == "--rollout-stop-on-existing-node"
        or argument.startswith(_ROLLOUT_CONFIG_CLI_PREFIXES)
        for argument in argv_list
    )
    parser = build_launcher_argument_parser()
    parsed = parser.parse_args(argv_list)
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
        require_exact_or_terminal=parsed.require_exact_or_terminal,
        min_depth=parsed.min_depth,
        min_visit_count=parsed.min_visit_count,
        max_rows=parsed.max_rows,
        use_backed_up_value=parsed.use_backed_up_value,
        evaluator_update_policy=parsed.evaluator_update_policy,
        pipeline_mode=parsed.pipeline_mode,
        training_export_mode=parsed.training_export_mode,
        training_evaluator_names=_parse_training_evaluator_names(
            parsed.training_evaluator_names
        ),
        training_max_rows=parsed.training_max_rows,
        training_row_chunk_size=parsed.training_row_chunk_size,
        skip_evaluator_diagnostics=parsed.skip_evaluator_diagnostics,
        evaluator_diagnostics_max_rows=parsed.evaluator_diagnostics_max_rows,
        growth_memory_profile=parsed.growth_memory_profile,
        growth_memory_profile_top_n=parsed.growth_memory_profile_top_n,
        growth_memory_profile_sample_nodes=parsed.growth_memory_profile_sample_nodes,
        candidate_checkpoint_load_headroom_factor=(
            parsed.candidate_checkpoint_load_headroom_factor
        ),
        candidate_checkpoint_load_min_headroom_mb=(
            parsed.candidate_checkpoint_load_min_headroom_mb
        ),
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
        min_available_ram_mb=parsed.min_available_ram_mb,
        tree_branch_limit=parsed.tree_branch_limit,
        reevaluation_blend_alpha=parsed.reevaluation_blend_alpha,
        search=MorpionBootstrapSearchConfig(
            rollout=MorpionBootstrapRolloutConfig(
                enabled=parsed.rollout_after_opening,
                max_extra_steps=parsed.rollout_max_extra_steps,
                action_selector_kind=parsed.rollout_action_selector_kind,
                random_seed=parsed.rollout_random_seed,
                stop_on_existing_node=parsed.rollout_stop_on_existing_node,
            )
        ),
    )
    return MorpionBootstrapLauncherArgs(
        bootstrap_args=bootstrap_args,
        max_cycles=parsed.max_cycles,
        pipeline_stage=cast("MorpionPipelineStage", parsed.pipeline_stage),
        pipeline_generation=parsed.pipeline_generation,
        reevaluation_max_nodes_per_patch=parsed.reevaluation_max_nodes_per_patch,
        verbose_checkpoint_logs=parsed.verbose_checkpoint_logs,
        training_export_mode_explicit=training_export_mode_explicit,
        rollout_config_explicit=rollout_config_explicit,
        min_available_ram_mb_explicit=min_available_ram_mb_explicit,
        tree_branch_limit_explicit=tree_branch_limit_explicit,
        candidate_checkpoint_load_headroom_explicit=(
            candidate_checkpoint_load_headroom_explicit
        ),
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
    _configure_anemone_checkpoint_logging(
        verbose_checkpoint_logs=launcher_args.verbose_checkpoint_logs
    )
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
