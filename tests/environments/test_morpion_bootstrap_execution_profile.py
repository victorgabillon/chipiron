"""Explicit bootstrap profile identities and refusal before authority mutation."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_from_dict,
    bootstrap_config_to_canonical_json,
    bootstrap_config_to_dict,
    save_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.execution_profile import (
    KEY,
    args_with_policy,
    policy_from_metadata,
    policy_metadata,
    require_workspace_profile,
)
from chipiron.environments.morpion.bootstrap.launcher import launcher_args_from_cli
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)

if TYPE_CHECKING:
    from pathlib import Path


def strict_args(root: Path) -> MorpionBootstrapArgs:
    """Declare the supported process recipe independently from the chosen device."""
    return MorpionBootstrapArgs(
        root,
        pipeline_mode="artifact_pipeline",
        training_device="cpu",
        execution_profile="strict_math",
        torch_intra_op_threads=1,
        torch_inter_op_threads=1,
    )


@pytest.mark.fast
def test_explicit_cli_config_roundtrip_and_absent_default_bytes(tmp_path: Path) -> None:
    """Default identity is unchanged; new explicit profile is immutable metadata."""
    args = MorpionBootstrapArgs(tmp_path)
    baseline = bootstrap_config_from_args(args)
    explicit_existing = bootstrap_config_from_args(
        replace(args, execution_profile="existing")
    )
    assert bootstrap_config_to_canonical_json(
        baseline
    ) == bootstrap_config_to_canonical_json(explicit_existing)
    assert KEY not in baseline.metadata
    parsed = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--scientific",
        "--scientific-operation",
        "prepare-continuous",
        "--training-device",
        "cuda:0",
        "--execution-profile",
        "strict_math",
        "--torch-intra-op-threads",
        "1",
        "--torch-inter-op-threads",
        "1",
    ])
    config = bootstrap_config_from_args(parsed.bootstrap_args)
    assert bootstrap_config_from_dict(bootstrap_config_to_dict(config)) == config
    restored = args_with_policy(args, config)
    assert policy_metadata(restored) == policy_metadata(parsed.bootstrap_args)
    assert restored.training_device == "cuda:0"
    with pytest.raises(ValueError, match="changed on resume"):
        args_with_policy(replace(args, execution_profile="existing"), config)
    with pytest.raises(ValueError, match="changed on resume"):
        args_with_policy(strict_args(tmp_path), baseline)


@pytest.mark.fast
@pytest.mark.parametrize(
    "change",
    [
        {"torch_intra_op_threads": 2},
        {"torch_inter_op_threads": 2},
        {"torch_intra_op_threads": True},
        {"execution_profile": "strict"},
        {"execution_profile": None},
    ],
)
def test_unsupported_or_incomplete_process_policy_rejected(
    tmp_path: Path, change: dict
) -> None:
    """No silent profile/precision/thread fallback is admitted to experiment config."""
    with pytest.raises(ValueError):
        bootstrap_config_from_args(replace(strict_args(tmp_path), **change))
    assert not list(tmp_path.iterdir())


@pytest.mark.fast
def test_malformed_policy_cannot_be_downgraded(tmp_path: Path) -> None:
    """Null, unknown keys and bool-as-integer records fail closed."""
    record = policy_metadata(strict_args(tmp_path))[KEY]
    assert isinstance(record, dict)
    for value in (
        None,
        {},
        {**record, "schema_version": True},
        {**record, "profile": "existing"},
        {**record, "extra": 1},
    ):
        with pytest.raises(ValueError):
            policy_from_metadata({KEY: value})


@pytest.mark.fast
@pytest.mark.parametrize(
    "entry",
    [
        "growth",
        "dataset",
        "training",
        "dataset_worker",
        "training_worker",
        "reevaluation",
        "orchestrator",
        "prepare",
        "legacy",
    ],
)
def test_manual_entries_refuse_unowned_strict_before_mutation(
    tmp_path: Path, entry: str
) -> None:
    """Manual stage callers cannot open claims or create artifacts without an owner."""
    from chipiron.environments.morpion.bootstrap.bootstrap_loop import (
        run_morpion_bootstrap_loop,
    )
    from chipiron.environments.morpion.bootstrap.continual_operator import (
        prepare_continual_experiment,
    )
    from chipiron.environments.morpion.bootstrap.pipeline.stages import (
        run_pipeline_dataset_stage,
        run_pipeline_growth_stage,
        run_pipeline_training_stage,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
        run_morpion_artifact_pipeline_once,
        run_next_pipeline_dataset_stage_once,
        run_next_pipeline_training_stage_once,
    )
    from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
        run_morpion_reevaluation_worker_once,
    )

    args = strict_args(tmp_path / "untouched")
    calls = {
        "growth": lambda: run_pipeline_growth_stage(args, object()),
        "dataset": lambda: run_pipeline_dataset_stage(args, generation=1),
        "training": lambda: run_pipeline_training_stage(args, generation=1),
        "dataset_worker": lambda: run_next_pipeline_dataset_stage_once(args),
        "training_worker": lambda: run_next_pipeline_training_stage_once(args),
        "reevaluation": lambda: run_morpion_reevaluation_worker_once(args),
        "orchestrator": lambda: run_morpion_artifact_pipeline_once(args, object()),
        "prepare": lambda: prepare_continual_experiment(
            args, MorpionContinualLearningPolicy()
        ),
        "legacy": lambda: run_morpion_bootstrap_loop(args, object()),
    }
    with pytest.raises(ValueError, match=r"bootstrap_execution|scientific artifact"):
        calls[entry]()
    assert not (tmp_path / "untouched").exists()


@pytest.mark.fast
@pytest.mark.parametrize(
    "entry",
    [
        "activate",
        "automatic",
        "prepare-activation",
        "monitor",
        "guard",
        "admit-training",
        "publish-training",
        "accept-training",
    ],
)
def test_workspace_entries_refuse_unowned_profile(tmp_path: Path, entry: str) -> None:
    """Persisted strict consumers require the declared owner even without CLI args."""
    from chipiron.environments.morpion.bootstrap import training_publication as training
    from chipiron.environments.morpion.bootstrap.continual_operator import (
        run_monitored_continual_experiment,
    )
    from chipiron.environments.morpion.bootstrap.model_activation import (
        activate_model,
        activate_selected_round_automatically,
        prepare_model_activation_request,
    )

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_config(
        bootstrap_config_from_args(strict_args(tmp_path)), paths.bootstrap_config_path
    )
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    calls = {
        "activate": lambda: activate_model(paths, {}),
        "automatic": lambda: activate_selected_round_automatically(paths),
        "prepare-activation": lambda: prepare_model_activation_request(
            paths, training_generation=1
        ),
        "monitor": lambda: run_monitored_continual_experiment(
            paths, tmp_path / "no-script"
        ),
        "guard": lambda: require_workspace_profile(paths),
        "admit-training": lambda: training.admit_training(
            paths, object(), {}, ttl_seconds=1, owner=None
        ).__enter__(),
        "publish-training": lambda: training.publish_training(
            paths, object(), object()
        ),
        "accept-training": lambda: training.accept_training(paths, {}),
    }
    with pytest.raises(ValueError, match="bootstrap_execution"):
        calls[entry]()
    assert before == {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }


@pytest.mark.fast
@pytest.mark.parametrize("failure", ["none", "error", "interrupt"])
def test_fresh_cli_process_owns_scope_and_restores_flags(
    tmp_path: Path, failure: str
) -> None:
    """Each interpreter configures inter-op once; the lifetime owns flags until exit."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        import torch
        from chipiron.environments.morpion.bootstrap import launcher
        from chipiron.learning.torch_runtime import torch_execution_profile_info, configure_torch_process_threads
        before = torch_execution_profile_info()
        def body(args):
            current = torch_execution_profile_info()
            assert current.effective == "strict_math"
            assert current.deterministic and not current.warn_only
            assert not current.mha_fastpath
            assert current.math_sdp and not current.flash_sdp
            assert current.intra_op_threads == current.inter_op_threads == 1
            if sys.argv[2] == "error": raise RuntimeError("body error")
            if sys.argv[2] == "interrupt": raise KeyboardInterrupt("body interrupt")
        launcher._run_launcher_cli = body
        try:
            launcher.main([
                "--work-dir", sys.argv[1], "--scientific",
                "--scientific-operation", "prepare-continuous",
                "--training-device", "cpu", "--execution-profile", "strict_math",
                "--torch-intra-op-threads", "1", "--torch-inter-op-threads", "1",
            ])
        except (RuntimeError, KeyboardInterrupt) as error:
            assert str(error) == "body " + sys.argv[2]
        else:
            assert sys.argv[2] == "none"
        after = torch_execution_profile_info()
        for name in ("effective", "deterministic", "warn_only", "mha_fastpath",
                     "math_sdp", "flash_sdp", "mem_efficient_sdp", "cudnn_sdp"):
            assert getattr(after, name) == getattr(before, name), name
        assert after.intra_op_threads == after.inter_op_threads == 1
        configure_torch_process_threads(intra_op=1, inter_op=1)
        try:
            configure_torch_process_threads(intra_op=2, inter_op=2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("late inter-op change was not rejected")
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), failure],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.fast
def test_inspect_does_not_enter_runtime_or_initialize_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only status on a configured GPU experiment works without that backend."""
    from chipiron.environments.morpion.bootstrap import execution_profile
    from chipiron.learning.torch_runtime import torch_execution_profile_info

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_config(
        bootstrap_config_from_args(
            replace(strict_args(tmp_path), training_device="cuda:0")
        ),
        paths.bootstrap_config_path,
    )
    before = torch_execution_profile_info()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("inspection must not configure runtime or resolve a GPU")

    monkeypatch.setattr(execution_profile, "configure_torch_process_threads", forbidden)
    monkeypatch.setattr(execution_profile, "resolve_torch_device", forbidden)
    launcher = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--scientific",
        "--scientific-operation",
        "inspect",
    ])
    with execution_profile.launcher_execution(launcher, process_start=True) as resolved:
        assert resolved.bootstrap_args.execution_profile == "strict_math"
        assert torch_execution_profile_info() == before
    assert torch_execution_profile_info() == before


@pytest.mark.fast
@pytest.mark.parametrize("mismatch", ["legacy", "build", "threads"])
def test_wrong_genesis_profile_refused_before_entry(
    tmp_path: Path, mismatch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persisted explicit config cannot adopt a legacy or incompatible genesis."""
    import torch

    from chipiron.environments.morpion.bootstrap.run_state import (
        initialize_bootstrap_run_state,
        save_bootstrap_run_state,
    )
    from chipiron.learning.torch_runtime import (
        TorchExecutionProfileError,
        torch_execution_profile,
        torch_execution_profile_metadata,
    )

    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_config(
        bootstrap_config_from_args(strict_args(tmp_path)), paths.bootstrap_config_path
    )
    with torch_execution_profile("strict_math"):
        metadata = (
            policy_metadata(strict_args(tmp_path)) | torch_execution_profile_metadata()
        )
        if mismatch == "legacy":
            metadata = {}
        else:
            record = metadata["torch_execution_profile"]
            record["torch_version" if mismatch == "build" else "intra_op_threads"] = (
                "incompatible" if mismatch == "build" else 2
            )
        save_bootstrap_run_state(
            replace(initialize_bootstrap_run_state(), metadata=metadata),
            paths.run_state_path,
        )
        before = paths.run_state_path.read_bytes()
        with pytest.raises((ValueError, TorchExecutionProfileError)):
            require_workspace_profile(paths)
        assert paths.run_state_path.read_bytes() == before


@pytest.mark.fast
def test_cli_preparation_checks_freshness_before_observation_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passive worker metrics cannot turn a fresh CLI preparation into a partial run."""
    from chipiron.environments.morpion.bootstrap import launcher

    root = tmp_path / "fresh"

    def prepare(_args: object) -> None:
        files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        assert files <= {"launcher.pid", "launcher_process_state.json"}

    monkeypatch.setattr(launcher, "run_morpion_bootstrap_experiment", prepare)
    assert (
        launcher.main([
            "--work-dir",
            str(root),
            "--scientific",
            "--scientific-operation",
            "prepare-continuous",
        ])
        == 0
    )
