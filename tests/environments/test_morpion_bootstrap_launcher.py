"""Tests for the canonical Morpion bootstrap launcher."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    save_training_tree_snapshot,
)
from atomheart.games.morpion import MorpionDynamics as AtomMorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

import chipiron.environments.morpion.bootstrap.launcher as launcher_module
from chipiron.environments.morpion.bootstrap import (
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    AnemoneMorpionSearchRunnerArgs,
    MorpionBootstrapArgs,
    MorpionBootstrapControl,
    MorpionBootstrapLauncherArgs,
    MorpionBootstrapPaths,
    MorpionBootstrapRunState,
    MorpionBootstrapRuntimeControl,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    initialize_bootstrap_run_state,
    run_morpion_bootstrap_experiment,
    save_bootstrap_config,
    save_bootstrap_control,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.config import bootstrap_config_from_args
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    canonical_morpion_evaluator_family_config,
)

if TYPE_CHECKING:
    import pytest


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_training_snapshot(
    *,
    target_value: float,
    root_node_id: str,
) -> TrainingTreeSnapshot:
    """Build one minimal valid training snapshot for launcher tests."""
    node = TrainingNodeSnapshot(
        node_id=root_node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=True,
        is_exact=True,
        over_event_label=None,
        visit_count=7,
        metadata={"source": "bootstrap-launcher-test"},
    )
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=(node,),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


class FakeMorpionSearchRunner:
    """Tiny deterministic runner satisfying the Morpion bootstrap protocol."""

    def __init__(
        self,
        *,
        tree_sizes: tuple[int, ...],
        target_values: tuple[float, ...],
    ) -> None:
        """Initialize the fake runner with per-cycle tree sizes and targets."""
        self._tree_sizes = tree_sizes
        self._target_values = target_values
        self._cycle_index = -1

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object | None = None,
    ) -> None:
        """Accept launcher loop restore inputs without side effects."""
        del tree_snapshot_path, model_bundle_path, effective_runtime_config

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner to the next predefined tree size."""
        del max_growth_steps
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Write one real training snapshot to ``output_path``."""
        index = max(self._cycle_index, 0)
        snapshot = _make_training_snapshot(
            target_value=self._target_values[index],
            root_node_id=f"node-{index}",
        )
        save_training_tree_snapshot(snapshot, output_path)

    def current_tree_size(self) -> int:
        """Return the current predefined tree size."""
        index = max(self._cycle_index, 0)
        return self._tree_sizes[index]


def _multi_evaluator_config() -> MorpionEvaluatorsConfig:
    """Return one representative two-evaluator bootstrap config."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec(
                name="linear",
                model_type="linear",
                hidden_sizes=None,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
            "mlp": MorpionEvaluatorSpec(
                name="mlp",
                model_type="mlp",
                hidden_sizes=(8, 4),
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
        }
    )


def _single_evaluator_config() -> MorpionEvaluatorsConfig:
    """Return one small single-evaluator config for real launcher loop tests."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec(
                name="linear",
                model_type="linear",
                hidden_sizes=None,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )
        }
    )


def _make_launcher_args(
    work_dir: Path,
    *,
    evaluators_config: MorpionEvaluatorsConfig | None = None,
    evaluator_family_preset: str | None = None,
    tree_branch_limit: int = 96,
    max_cycles: int | None = 1,
    open_dashboard: bool = False,
    print_startup_summary: bool = False,
    print_dashboard_hint: bool = False,
) -> MorpionBootstrapLauncherArgs:
    """Build representative launcher args for tests."""
    bootstrap_args = MorpionBootstrapArgs(
        work_dir=work_dir,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.5,
        save_after_seconds=10.0,
        max_rows=17,
        use_backed_up_value=False,
        tree_branch_limit=tree_branch_limit,
        batch_size=1,
        num_epochs=1,
        learning_rate=1e-3,
        shuffle=False,
        evaluators_config=evaluators_config,
        evaluator_family_preset=evaluator_family_preset,
    )
    return MorpionBootstrapLauncherArgs(
        bootstrap_args=bootstrap_args,
        max_cycles=max_cycles,
        open_dashboard=open_dashboard,
        print_startup_summary=print_startup_summary,
        print_dashboard_hint=print_dashboard_hint,
    )


def test_fresh_run_startup_summary_reports_expected_state(tmp_path: Path) -> None:
    """Fresh launcher summary should show a new run and canonical paths."""
    launcher_args = _make_launcher_args(
        tmp_path,
        evaluators_config=_multi_evaluator_config(),
        max_cycles=0,
    )

    startup_status = launcher_module._collect_launcher_startup_status(launcher_args)
    summary = launcher_module._render_launcher_startup_summary(
        startup_status,
        dashboard_requested=False,
    )

    assert "mode: fresh run" in summary
    assert "bootstrap config: absent; will be written from launcher args" in summary
    assert "run state: absent" in summary
    assert "history: absent" in summary
    assert f"work dir: {tmp_path.resolve()}" in summary
    assert (
        f"config: {MorpionBootstrapPaths.from_work_dir(tmp_path).bootstrap_config_path}"
        in summary
    )


def test_resume_startup_summary_reports_resume_state(tmp_path: Path) -> None:
    """Resume summary should surface persisted artifacts and latest indices."""
    launcher_args = _make_launcher_args(
        tmp_path,
        evaluators_config=_multi_evaluator_config(),
        max_cycles=0,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_config(
        bootstrap_config_from_args(launcher_args.bootstrap_args),
        paths.bootstrap_config_path,
    )
    save_bootstrap_control(
        MorpionBootstrapControl(
            force_evaluator="linear",
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64),
        ),
        paths.control_path,
    )
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=3,
            cycle_index=7,
            latest_tree_snapshot_path="tree_exports/generation_000003.json",
            latest_rows_path="rows/generation_000003.json",
            latest_model_bundle_paths={"linear": "models/generation_000003/linear"},
            active_evaluator_name="linear",
            tree_size_at_last_save=42,
            last_save_unix_s=123.0,
        ),
        paths.run_state_path,
    )
    paths.history_jsonl_path.write_text("", encoding="utf-8")

    startup_status = launcher_module._collect_launcher_startup_status(launcher_args)
    summary = launcher_module._render_launcher_startup_summary(
        startup_status,
        dashboard_requested=True,
    )

    assert "mode: resume" in summary
    assert "bootstrap config: present" in summary
    assert "control file: present" in summary
    assert "history: present" in summary
    assert "latest generation: 3" in summary
    assert "latest cycle: 7" in summary
    assert "forced evaluator control: linear" in summary
    assert "tree_branch_limit: 64 (baseline 96, control override 64)" in summary


def test_launcher_creates_artifacts_and_returns_final_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical launcher should create bootstrap artifacts and return run state."""
    launcher_args = _make_launcher_args(
        tmp_path,
        evaluators_config=_single_evaluator_config(),
    )
    created_runner_args: list[AnemoneMorpionSearchRunnerArgs] = []

    def _fake_runner_constructor(
        runner_args: AnemoneMorpionSearchRunnerArgs,
    ) -> FakeMorpionSearchRunner:
        created_runner_args.append(runner_args)
        return FakeMorpionSearchRunner(tree_sizes=(3,), target_values=(1.0,))

    monkeypatch.setattr(
        launcher_module,
        "AnemoneMorpionSearchRunner",
        _fake_runner_constructor,
    )

    run_state = run_morpion_bootstrap_experiment(launcher_args)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    assert created_runner_args
    assert paths.bootstrap_config_path.is_file()
    assert paths.run_state_path.is_file()
    assert paths.history_jsonl_path.is_file()
    assert isinstance(run_state, MorpionBootstrapRunState)
    assert run_state.generation == 1
    assert run_state.cycle_index == 0


def test_dashboard_hint_includes_exact_command(tmp_path: Path) -> None:
    """Dashboard hint should point operators at the supported dashboard command."""
    hint = launcher_module._render_dashboard_hint(
        tmp_path.resolve(),
        requested_open=True,
    )

    assert "Dashboard requested" in hint
    assert "python -m chipiron.environments.morpion.bootstrap.dashboard_app" in hint
    assert str(tmp_path.resolve()) in hint


def test_launcher_cli_main_parses_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI main should parse launcher args and dispatch the canonical entrypoint."""
    captured_args: list[MorpionBootstrapLauncherArgs] = []

    def _fake_run(
        launcher_args: MorpionBootstrapLauncherArgs,
    ) -> MorpionBootstrapRunState:
        captured_args.append(launcher_args)
        return initialize_bootstrap_run_state()

    monkeypatch.setattr(
        launcher_module,
        "run_morpion_bootstrap_experiment",
        _fake_run,
    )

    exit_code = launcher_module.main(
        [
            "--work-dir",
            str(tmp_path),
            "--evaluator-family",
            CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
            "--max-cycles",
            "2",
            "--dashboard",
            "--max-growth-steps-per-cycle",
            "12",
            "--save-after-seconds",
            "22.5",
            "--save-after-tree-growth-factor",
            "1.5",
            "--max-rows",
            "33",
            "--no-use-backed-up-value",
            "--tree-branch-limit",
            "48",
        ]
    )

    assert exit_code == 0
    assert len(captured_args) == 1
    launcher_args = captured_args[0]
    assert launcher_args.max_cycles == 2
    assert launcher_args.open_dashboard is True
    assert (
        launcher_args.bootstrap_args.evaluator_family_preset
        == CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
    )
    assert launcher_args.bootstrap_args.max_growth_steps_per_cycle == 12
    assert launcher_args.bootstrap_args.save_after_seconds == 22.5
    assert launcher_args.bootstrap_args.save_after_tree_growth_factor == 1.5
    assert launcher_args.bootstrap_args.max_rows == 33
    assert launcher_args.bootstrap_args.use_backed_up_value is False
    assert launcher_args.bootstrap_args.tree_branch_limit == 48


def test_launcher_constructs_real_runner_in_normal_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical launcher should own real runner construction before loop dispatch."""
    launcher_args = _make_launcher_args(
        tmp_path,
        evaluators_config=_single_evaluator_config(),
        tree_branch_limit=40,
        max_cycles=3,
    )
    sentinel_runner = object()
    created_runner_args: list[AnemoneMorpionSearchRunnerArgs] = []

    def _fake_runner_constructor(
        runner_args: AnemoneMorpionSearchRunnerArgs,
    ) -> object:
        created_runner_args.append(runner_args)
        return sentinel_runner

    def _fake_loop(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_cycles: int | None = None,
    ) -> MorpionBootstrapRunState:
        assert args == launcher_module._resolve_launcher_bootstrap_args(launcher_args)
        assert runner is sentinel_runner
        assert max_cycles == 3
        return initialize_bootstrap_run_state()

    monkeypatch.setattr(
        launcher_module,
        "AnemoneMorpionSearchRunner",
        _fake_runner_constructor,
    )
    monkeypatch.setattr(launcher_module, "run_morpion_bootstrap_loop", _fake_loop)

    run_state = run_morpion_bootstrap_experiment(launcher_args)

    assert run_state == initialize_bootstrap_run_state()
    assert len(created_runner_args) == 1
    stopping_criterion = created_runner_args[0].search_args.stopping_criterion
    assert stopping_criterion.tree_branch_limit == 40


def test_startup_summary_renders_requested_evaluator_family_preset(
    tmp_path: Path,
) -> None:
    """Startup summary should show the selected evaluator family preset when used."""
    launcher_args = MorpionBootstrapLauncherArgs(
        bootstrap_args=MorpionBootstrapArgs(
            work_dir=tmp_path,
            evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
        ),
        print_startup_summary=False,
        print_dashboard_hint=False,
    )

    startup_status = launcher_module._collect_launcher_startup_status(launcher_args)
    summary = launcher_module._render_launcher_startup_summary(
        startup_status,
        dashboard_requested=False,
    )

    assert (
        "evaluator family preset: "
        f"{CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET} (explicit)" in summary
    )


def test_launcher_defaults_canonical_family_when_unset(tmp_path: Path) -> None:
    """Launcher-only defaulting should inject the canonical evaluator family preset."""
    launcher_args = _make_launcher_args(tmp_path, evaluators_config=None)

    resolved_args = launcher_module._resolve_launcher_bootstrap_args(launcher_args)
    startup_status = launcher_module._collect_launcher_startup_status(launcher_args)

    assert launcher_args.bootstrap_args.evaluator_family_preset is None
    assert (
        resolved_args.evaluator_family_preset
        == CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
    )
    assert startup_status.evaluator_family_source == "launcher_default"
    assert (
        startup_status.resolved_evaluator_family_preset
        == CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
    )
    assert (
        startup_status.bootstrap_config.evaluators
        == canonical_morpion_evaluator_family_config()
    )


def test_launcher_preserves_explicit_evaluator_family_preset(tmp_path: Path) -> None:
    """Explicit launcher preset should not be relabeled as a launcher default."""
    launcher_args = _make_launcher_args(
        tmp_path,
        evaluators_config=None,
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )

    startup_status = launcher_module._collect_launcher_startup_status(launcher_args)

    assert startup_status.evaluator_family_source == "explicit"
    assert (
        startup_status.resolved_evaluator_family_preset
        == CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
    )


def test_launcher_explicit_evaluator_config_suppresses_default_family(
    tmp_path: Path,
) -> None:
    """Explicit evaluator config should prevent launcher-family injection."""
    explicit_config = _single_evaluator_config()
    launcher_args = _make_launcher_args(
        tmp_path,
        evaluators_config=explicit_config,
    )

    resolved_args = launcher_module._resolve_launcher_bootstrap_args(launcher_args)
    startup_status = launcher_module._collect_launcher_startup_status(launcher_args)
    summary = launcher_module._render_launcher_startup_summary(
        startup_status,
        dashboard_requested=False,
    )

    assert resolved_args.evaluator_family_preset is None
    assert startup_status.evaluator_family_source == "explicit_config"
    assert startup_status.resolved_evaluator_family_preset is None
    assert startup_status.bootstrap_config.evaluators == explicit_config
    assert "evaluator family preset: none (explicit evaluators_config)" in summary


def test_launcher_loop_receives_resolved_canonical_family_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical launcher path should pass resolved default family args into the loop."""
    launcher_args = _make_launcher_args(
        tmp_path,
        evaluators_config=None,
        max_cycles=2,
    )
    captured_args: list[MorpionBootstrapArgs] = []

    def _fake_runner_constructor(
        runner_args: AnemoneMorpionSearchRunnerArgs,
    ) -> object:
        del runner_args
        return object()

    def _fake_loop(
        args: MorpionBootstrapArgs,
        runner: object,
        *,
        max_cycles: int | None = None,
    ) -> MorpionBootstrapRunState:
        del runner
        captured_args.append(args)
        assert max_cycles == 2
        return initialize_bootstrap_run_state()

    monkeypatch.setattr(
        launcher_module,
        "AnemoneMorpionSearchRunner",
        _fake_runner_constructor,
    )
    monkeypatch.setattr(launcher_module, "run_morpion_bootstrap_loop", _fake_loop)

    run_morpion_bootstrap_experiment(launcher_args)

    assert len(captured_args) == 1
    assert (
        captured_args[0].resolved_evaluators_config()
        == canonical_morpion_evaluator_family_config()
    )


def test_startup_summary_marks_launcher_default_family(tmp_path: Path) -> None:
    """Startup summary should make launcher-default family provenance obvious."""
    launcher_args = _make_launcher_args(tmp_path, evaluators_config=None)

    startup_status = launcher_module._collect_launcher_startup_status(launcher_args)
    summary = launcher_module._render_launcher_startup_summary(
        startup_status,
        dashboard_requested=False,
    )

    assert (
        "evaluator family preset: "
        f"{CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET} (launcher default)" in summary
    )
    assert (
        "configured evaluators: linear_10, linear_20, linear_41, linear_5, "
        "mlp_10, mlp_20, mlp_41, mlp_5" in summary
    )


def test_direct_bootstrap_args_keep_legacy_default_behavior(tmp_path: Path) -> None:
    """Direct low-level bootstrap args should retain the old single-evaluator fallback."""
    bootstrap_args = MorpionBootstrapArgs(work_dir=tmp_path)

    assert bootstrap_args.evaluator_family_preset is None
    assert tuple(bootstrap_args.resolved_evaluators_config().evaluators) == ("default",)


def test_launcher_main_registers_and_marks_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launcher main should register process ownership on entry and mark stopped on exit."""
    calls: list[str] = []

    def _fake_register(paths: MorpionBootstrapPaths) -> object:
        calls.append(f"register:{paths.work_dir}")
        return object()

    def _fake_mark(
        paths: MorpionBootstrapPaths,
        *,
        exit_code: int | None = None,
        reason: str = "launcher_exit",
    ) -> object:
        calls.append(f"mark:{paths.work_dir}:{exit_code}:{reason}")
        return object()

    monkeypatch.setattr(
        launcher_module, "register_current_launcher_process", _fake_register
    )
    monkeypatch.setattr(
        launcher_module, "mark_current_launcher_process_stopped", _fake_mark
    )
    monkeypatch.setattr(
        launcher_module,
        "run_morpion_bootstrap_experiment",
        lambda launcher_args: initialize_bootstrap_run_state(),
    )

    exit_code = launcher_module.main(["--work-dir", str(tmp_path), "--max-cycles", "0"])

    assert exit_code == 0
    assert calls == [
        f"register:{tmp_path.resolve()}",
        f"mark:{tmp_path.resolve()}:0:launcher_exit",
    ]
