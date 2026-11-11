"""Fresh Morpion epochs and saved recipe authority across launcher restarts."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_loop import (
    run_morpion_bootstrap_loop,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import (
    IncompatibleStageBootstrapConfigError,
    bootstrap_config_from_args,
    bootstrap_config_from_dict,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
    load_bootstrap_config,
    save_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    MorpionBootstrapLauncherArgs,
    _collect_launcher_startup_status,
    launcher_args_from_cli,
    run_morpion_bootstrap_experiment,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
    save_bootstrap_run_state,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.search_runner_protocol import (
        MorpionSearchRunner,
    )


@pytest.mark.fast
def test_fresh_single_and_nine_candidate_family_resolve_one_epoch(
    tmp_path: Path,
) -> None:
    """The public fresh launcher persists one epoch for every generated member."""
    single = bootstrap_config_from_args(MorpionBootstrapArgs(work_dir=tmp_path))
    assert [spec.num_epochs for spec in single.evaluators.evaluators.values()] == [1]

    launch = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path / "family"),
        "--evaluator-family",
        CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    ])
    startup = _collect_launcher_startup_status(launch)
    specs = startup.bootstrap_config.evaluators.evaluators
    assert len(specs) == 9
    assert specs["transformer_v1"].num_epochs == 1
    assert {spec.num_epochs for spec in specs.values()} == {1}
    assert (
        load_bootstrap_config(startup.paths.bootstrap_config_path)
        == startup.bootstrap_config
    )
    assert (
        bootstrap_config_from_dict(bootstrap_config_to_dict(startup.bootstrap_config))
        == startup.bootstrap_config
    )


@pytest.mark.fast
def test_explicit_mixed_epochs_and_saved_five_survive_default_change(
    tmp_path: Path,
) -> None:
    """Explicit recipe values and an existing saved five-epoch family stay intact."""
    family_args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        evaluator_family_preset=CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    )
    fresh = family_args.resolved_evaluators_config().evaluators
    mixed = dict(fresh)
    mixed["transformer_v1"] = replace(fresh["transformer_v1"], num_epochs=5)
    mixed_config = bootstrap_config_from_args(
        replace(
            family_args,
            evaluator_family_preset=None,
            evaluators_config=MorpionEvaluatorsConfig(evaluators=mixed),
        )
    )
    assert mixed_config.evaluators.evaluators["transformer_v1"].num_epochs == 5
    assert {
        spec.num_epochs for spec in mixed_config.evaluators.evaluators.values()
    } == {1, 5}

    old = bootstrap_config_from_args(
        replace(
            family_args,
            evaluator_family_preset=None,
            evaluators_config=MorpionEvaluatorsConfig(
                evaluators={
                    name: replace(spec, num_epochs=5) for name, spec in fresh.items()
                }
            ),
        )
    )
    config_path = tmp_path / "bootstrap_config.json"
    save_bootstrap_config(old, config_path)
    original_hash = bootstrap_config_sha256(old)
    launch = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--evaluator-family",
        CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    ])
    startup = _collect_launcher_startup_status(launch)
    assert startup.bootstrap_config == old
    assert {
        spec.num_epochs
        for spec in startup.resolved_bootstrap_args.resolved_evaluators_config().evaluators.values()
    } == {5}
    assert bootstrap_config_sha256(load_bootstrap_config(config_path)) == original_hash

    run_morpion_bootstrap_loop(
        MorpionBootstrapArgs(
            work_dir=tmp_path,
            evaluator_family_preset=CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
        ),
        cast("MorpionSearchRunner", object()),
        max_cycles=0,
    )
    assert bootstrap_config_sha256(load_bootstrap_config(config_path)) == original_hash


@pytest.mark.fast
def test_legacy_preset_only_run_keeps_historical_five(tmp_path: Path) -> None:
    """An existing pre-config-file run is not reinterpreted as a fresh recipe."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_run_state(initialize_bootstrap_run_state(), paths.run_state_path)
    launch = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--evaluator-family",
        CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    ])
    startup = _collect_launcher_startup_status(launch)
    assert {
        spec.num_epochs
        for spec in startup.bootstrap_config.evaluators.evaluators.values()
    } == {5}
    assert {
        spec.num_epochs
        for spec in startup.resolved_bootstrap_args.resolved_evaluators_config().evaluators.values()
    } == {5}


@pytest.mark.parametrize("history", ["fresh", "saved", "legacy"])
@pytest.mark.parametrize("epochs", [None, 1, 3, 5])
def test_direct_loop_preserves_explicit_epoch_intent(
    tmp_path: Path, history: str, epochs: int | None
) -> None:
    """Public legacy loop honors explicit one/three/five; omission preserves history."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    saved = bootstrap_config_from_args(MorpionBootstrapArgs(tmp_path, num_epochs=5))
    if history == "saved":
        save_bootstrap_config(saved, paths.bootstrap_config_path)
    elif history == "legacy":
        save_bootstrap_run_state(initialize_bootstrap_run_state(), paths.run_state_path)
    args = MorpionBootstrapArgs(tmp_path, num_epochs=epochs)
    # Runtime copies must not turn omission into an explicit one.
    args = replace(args)
    run_morpion_bootstrap_loop(
        args, cast("MorpionSearchRunner", object()), max_cycles=0
    )
    actual = load_bootstrap_config(paths.bootstrap_config_path)
    expected = epochs if epochs is not None else (1 if history == "fresh" else 5)
    assert actual.evaluators.evaluators["default"].num_epochs == expected
    if expected == 5:
        assert bootstrap_config_sha256(actual) == bootstrap_config_sha256(saved)


def _run_config_only_launcher(
    args: MorpionBootstrapArgs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the public launcher with zero cycles and an unused search runner."""
    from chipiron.environments.morpion.bootstrap import launcher

    monkeypatch.setattr(launcher, "_build_launcher_runner", lambda _status: object())
    run_morpion_bootstrap_experiment(
        MorpionBootstrapLauncherArgs(
            bootstrap_args=args,
            max_cycles=0,
            print_startup_summary=False,
            print_dashboard_hint=False,
        )
    )


@pytest.mark.parametrize("epochs", [1, 3])
def test_public_launcher_refuses_explicit_saved_epoch_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, epochs: int
) -> None:
    """Stage compatibility rejects explicit recipes without changing saved bytes."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    args = MorpionBootstrapArgs(tmp_path, num_epochs=5)
    saved = bootstrap_config_from_args(args)
    save_bootstrap_config(saved, paths.bootstrap_config_path)
    original = paths.bootstrap_config_path.read_bytes()
    explicit = MorpionEvaluatorsConfig(
        evaluators={
            "default": replace(
                saved.evaluators.evaluators["default"], num_epochs=epochs
            )
        }
    )
    with pytest.raises(IncompatibleStageBootstrapConfigError, match="evaluators"):
        _run_config_only_launcher(
            replace(args, evaluators_config=explicit), monkeypatch
        )
    assert paths.bootstrap_config_path.read_bytes() == original
    assert bootstrap_config_sha256(
        load_bootstrap_config(paths.bootstrap_config_path)
    ) == bootstrap_config_sha256(saved)


@pytest.mark.parametrize("entry", ["direct", "launcher"])
def test_public_entries_preserve_explicit_mixed_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """Per-evaluator values take precedence and preserve their round-trip identity."""
    base = (
        MorpionBootstrapArgs(tmp_path)
        .resolved_evaluators_config()
        .evaluators["default"]
    )
    explicit = MorpionEvaluatorsConfig(
        evaluators={
            "first": replace(base, name="first", num_epochs=1),
            "second": replace(base, name="second", num_epochs=3),
            "third": replace(base, name="third", num_epochs=5),
        }
    )
    args = MorpionBootstrapArgs(tmp_path, num_epochs=7, evaluators_config=explicit)
    expected = bootstrap_config_from_args(args)
    for _ in range(2):
        if entry == "direct":
            run_morpion_bootstrap_loop(
                args, cast("MorpionSearchRunner", object()), max_cycles=0
            )
        else:
            _run_config_only_launcher(args, monkeypatch)
        actual = load_bootstrap_config(
            MorpionBootstrapPaths.from_work_dir(tmp_path).bootstrap_config_path
        )
        assert actual.evaluators == explicit
        assert bootstrap_config_sha256(actual) == bootstrap_config_sha256(expected)


@pytest.mark.parametrize("history", ["fresh", "saved", "legacy"])
@pytest.mark.parametrize("entry", ["direct", "launcher"])
def test_named_family_keeps_precedence_over_explicit_single_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: str, entry: str
) -> None:
    """A top-level three does not become a new global family override."""
    args = MorpionBootstrapArgs(
        tmp_path,
        num_epochs=3,
        evaluator_family_preset=CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    saved = bootstrap_config_from_args(args)
    saved = replace(
        saved,
        evaluators=MorpionEvaluatorsConfig(
            evaluators={
                name: replace(spec, num_epochs=5)
                for name, spec in saved.evaluators.evaluators.items()
            }
        ),
    )
    if history == "saved":
        save_bootstrap_config(saved, paths.bootstrap_config_path)
    elif history == "legacy":
        save_bootstrap_run_state(initialize_bootstrap_run_state(), paths.run_state_path)
    if entry == "direct":
        run_morpion_bootstrap_loop(
            args, cast("MorpionSearchRunner", object()), max_cycles=0
        )
    else:
        _run_config_only_launcher(args, monkeypatch)
    actual = load_bootstrap_config(paths.bootstrap_config_path)
    assert {spec.num_epochs for spec in actual.evaluators.evaluators.values()} == (
        {1} if history == "fresh" else {5}
    )
    if history != "fresh":
        assert bootstrap_config_sha256(actual) == bootstrap_config_sha256(saved)


@pytest.mark.parametrize("epochs", [1, 3])
@pytest.mark.parametrize("per_evaluator", [False, True])
@pytest.mark.parametrize("operation", ["worker", "initialize"])
def test_scientific_public_entry_rejects_explicit_epoch_conflict(
    tmp_path: Path, epochs: int, *, per_evaluator: bool, operation: str
) -> None:
    """Scientific adoption never hides explicit epochs, including explicit one."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    saved_args = MorpionBootstrapArgs(
        tmp_path,
        num_epochs=5,
        pipeline_mode="artifact_pipeline",
        runtime_checkpoint_format="sharded",
        training_export_mode="sharded",
    )
    saved = bootstrap_config_from_args(saved_args)
    save_bootstrap_config(saved, paths.bootstrap_config_path)
    save_bootstrap_run_state(initialize_bootstrap_run_state(), paths.run_state_path)
    original = paths.bootstrap_config_path.read_bytes()
    state_bytes = paths.run_state_path.read_bytes()
    requested = replace(saved_args, num_epochs=epochs)
    if per_evaluator:
        requested = replace(
            requested, evaluators_config=requested.resolved_evaluators_config()
        )
    launch = MorpionBootstrapLauncherArgs(
        bootstrap_args=requested,
        scientific=True,
        pipeline_stage="training_worker",
        scientific_operation="initialize" if operation == "initialize" else None,
        scientific_config=paths.bootstrap_config_path,
        seed_model=tmp_path / "unread-seed",
        print_startup_summary=False,
        print_dashboard_hint=False,
    )
    with pytest.raises(
        IncompatibleStageBootstrapConfigError,
        match=r"evaluators.default.num_epochs.*persisted=5, requested=" + str(epochs),
    ):
        run_morpion_bootstrap_experiment(launch)
    assert paths.bootstrap_config_path.read_bytes() == original
    assert paths.run_state_path.read_bytes() == state_bytes


@pytest.mark.parametrize("epochs", [None, 5])
def test_scientific_public_worker_keeps_saved_epochs_without_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, epochs: int | None
) -> None:
    """Omitted or matching explicit epochs reach execution with the saved identity."""
    from chipiron.environments.morpion.bootstrap import generation_publication_io

    class ScientificExecutionBoundaryError(Exception):
        """Stop before any scientific execution in this configuration-only test."""

    def stop_before_execution(*_args: object) -> None:
        raise ScientificExecutionBoundaryError

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    args = MorpionBootstrapArgs(
        tmp_path,
        num_epochs=5,
        pipeline_mode="artifact_pipeline",
        runtime_checkpoint_format="sharded",
        training_export_mode="sharded",
    )
    save_bootstrap_config(bootstrap_config_from_args(args), paths.bootstrap_config_path)
    save_bootstrap_run_state(initialize_bootstrap_run_state(), paths.run_state_path)
    original = paths.bootstrap_config_path.read_bytes()
    monkeypatch.setattr(
        generation_publication_io, "committed_evaluator", stop_before_execution
    )
    with pytest.raises(ScientificExecutionBoundaryError):
        run_morpion_bootstrap_experiment(
            MorpionBootstrapLauncherArgs(
                bootstrap_args=replace(args, num_epochs=epochs),
                scientific=True,
                pipeline_stage="training_worker",
                print_startup_summary=False,
                print_dashboard_hint=False,
            )
        )
    assert paths.bootstrap_config_path.read_bytes() == original
