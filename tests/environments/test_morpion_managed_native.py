"""Actual installed two-session monitor/continuation and matched bounded control."""

import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from anemone.checkpoints.rng_state import rollout_random_generator

from chipiron.environments.morpion.bootstrap import managed_sessions as managed
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import (
    MorpionBootstrapRolloutConfig,
    MorpionBootstrapSearchConfig,
)
from chipiron.environments.morpion.bootstrap.continual_operator import (
    prepare_continual_experiment,
)
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.execution_profile import (
    bootstrap_execution,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.test_morpion_checkpoint_parity import _selection_signature
from tests.environments.test_morpion_committed_evaluator_restore import fingerprint
from tests.environments.test_morpion_lifecycle_continuation import equal_state


def initialize(root: Path) -> MorpionBootstrapArgs:
    """Fresh two-candidate strict CPU genesis; each arm has the identical recipe."""
    args = MorpionBootstrapArgs(
        work_dir=root,
        training_device="cpu",
        execution_profile="strict_math",
        torch_intra_op_threads=1,
        torch_inter_op_threads=1,
        max_growth_steps_per_cycle=2,
        max_rows=128,
        validation_fraction=0.25,
        validation_seed=17,
        search=MorpionBootstrapSearchConfig(
            rollout=MorpionBootstrapRolloutConfig(
                enabled=True, max_extra_steps=2, random_seed=19
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
                for name, rate in [("linear_a", 0.001), ("linear_b", 0.003)]
            }
        ),
    )
    policy = MorpionContinualLearningPolicy(
        max_elapsed_seconds=12,
        max_tree_nodes=5000,
        min_available_ram_mb=1,
        shutdown_grace_seconds=5,
    )
    with bootstrap_execution(replace(args, continual_learning=policy)):
        prepare_continual_experiment(args, policy, managed_sessions=True)
    return replace(args, continual_learning=policy)


def invoke(
    root: Path, script: Path, rounds: int, *options: str
) -> subprocess.CompletedProcess[str]:
    """Use the real installed normal CLI and its fresh process thread ownership."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "chipiron.environments.morpion.bootstrap.operator_cli",
            "start",
            "--work-dir",
            str(root),
            "--cluster-script",
            str(script),
            *options,
        ],
        env=dict(
            os.environ,
            J3252_FIXTURE_ROUNDS=str(rounds),
            CHIPIRON_CANDIDATE_WORKERS="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        ),
        capture_output=True,
        text=True,
        timeout=40,
    )
    (root.parent / (root.name + "-last-cli.log")).write_text(
        result.stdout + result.stderr
    )
    return result


def scientific_result(args: MorpionBootstrapArgs) -> Any:
    """Compare tensors, Adam, complete fixed population, selection and next Growth."""
    from chipiron.environments.morpion.bootstrap.model_activation import _runner

    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    state = load_bootstrap_run_state(paths.run_state_path)
    records = [r for r in publication_records(state.metadata) if training.ACCEPTED in r]
    assert len(records) == 3
    previous = None
    rounds = []
    for record in records:
        proposal = training._proposal(paths, record[training.ACCEPTED])
        assert all(
            parent is None
            if previous is None
            else parent["accepted_training"] == previous
            for parent in proposal["admission"]["parents"].values()
        )
        models = {}
        for name, ref in proposal["models"].items():
            bundle = paths.work_dir / ref["path"]
            metadata = json.loads((bundle / "morpion_manifest.json").read_text())[
                "metadata"
            ]
            evidence = metadata["continuation"]
            models[name] = {
                "weights": torch.load(
                    bundle / "param.pt", map_location="cpu", weights_only=True
                ),
                "adam": torch.load(
                    bundle / "optimizer.pt", map_location="cpu", weights_only=True
                ),
                "seed": evidence["seed"],
                "starting_steps": evidence["starting_optimizer_step_count"],
                "final_steps": evidence["final_optimizer_step_count"],
                "loss": proposal["status"]["evaluator_results"][name][
                    "validation_loss"
                ],
            }
        decision = proposal["status"]["metadata"]["continual_selection"]
        rounds.append((
            models,
            proposal["status"]["selected_evaluator_name"],
            decision["validation_population"],
        ))
        previous = record[training.ACCEPTED]
    with bootstrap_execution(args):
        model, _ = committed_evaluator(paths.work_dir, state)
        runner, effective = _runner(paths, state)
        try:
            assert state.latest_runtime_checkpoint_path is not None
            runner.load_or_create(
                paths.work_dir / state.latest_runtime_checkpoint_path, None, effective
            )
            runner.grow(1)
            rollout_rng = rollout_random_generator(runner._runtime)
            assert rollout_rng is not None
            growth = (
                fingerprint(runner),
                _selection_signature(runner),
                repr(runner._random_generator.getstate()),
                repr(rollout_rng.getstate()),
            )
        finally:
            runner.discard_uncommitted_runtime()
    assert state.active_evaluator_name not in {
        "linear_a",
        "linear_b",
    }  # accepted is not activated
    return rounds, growth, model["sha256"]


@pytest.mark.exhaustive
def test_real_two_session_grant_keeps_loser_adam_and_scientific_suffix(
    tmp_path: Path,
) -> None:
    """Run strict numerical parity in a fresh process with declared threads."""
    code = (
        "import sys, torch; "
        "torch.set_num_threads(1); torch.set_num_interop_threads(1); "
        "sys.path.insert(0, sys.argv[1]); "
        "from pathlib import Path; "
        "from tests.environments.test_morpion_managed_native import "
        "_run_real_two_session_parity; "
        "_run_real_two_session_parity(Path(sys.argv[2]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(Path(__file__).resolve().parents[2]),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_real_two_session_parity(tmp_path: Path) -> None:
    """Fixed-work parity, not a time-defined trajectory or performance experiment."""
    import chipiron

    assert Path(chipiron.__file__).is_relative_to(Path(sys.prefix))
    helper = Path(__file__).with_name("managed_session_workload.py")
    # This bounded test supervisor delegates work to existing APIs. Separate existing
    # J317 fixtures exercise the production shell control/ACK and early-deadline path.
    script = tmp_path / "fixture-supervisor.sh"
    script.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " -B "
        + shlex.quote(str(helper))
        + ' "$MORPION_WORK_DIR"\n'
    )
    results = {}
    for label in ["control", "resumed"]:
        root = tmp_path / label
        args = initialize(root)
        first = invoke(root, script, 3 if label == "control" else 1)
        assert first.returncode == 0, first.stdout + first.stderr
        assert (root.parent / (root.name + "-work-done.json")).exists()
        state = load_bootstrap_run_state(root / "run_state.json")
        data = state.metadata[managed.KEY]
        assert (
            data["active"] is None
            and len(data["sessions"]) == 1
            and data["consumed_seconds"] >= 12
        )
        before_science = {k: v for k, v in state.metadata.items() if k != managed.KEY}
        if label == "resumed":
            second = invoke(
                root, script, 2, "--add-time", "12", "--request-id", "second-period"
            )
            assert second.returncode == 0, second.stdout + second.stderr
            state = load_bootstrap_run_state(root / "run_state.json")
            data = state.metadata[managed.KEY]
            assert (
                len(data["grants"]) == 1
                and len(data["sessions"]) == 2
                and data["authorized_seconds"] == 24
                and data["consumed_seconds"] >= 24
            )
            before = (root / "run_state.json").read_bytes()
            duplicate = invoke(
                root, script, 2, "--add-time", "12", "--request-id", "second-period"
            )
            assert (
                duplicate.returncode == 0
                and (root / "run_state.json").read_bytes() == before
            )
            assert len(state.metadata[managed.KEY]["sessions"]) == 2
        assert (
            before_science["bootstrap_config_hash"]
            == state.metadata["bootstrap_config_hash"]
        )
        results[label] = scientific_result(args)
        if label == "resumed":
            paths = MorpionBootstrapPaths.from_work_dir(root)
            accepted = publication_records(state.metadata)[-1][training.ACCEPTED]
            proposal = training._proposal(paths, accepted)
            parent = root / next(iter(proposal["models"].values()))["path"]
            (parent / "optimizer.pt").unlink()
            before = (root / "run_state.json").read_bytes()
            missing = invoke(
                root, script, 1, "--add-time", "12", "--request-id", "missing-adam"
            )
            assert (
                missing.returncode == 2
                and before == (root / "run_state.json").read_bytes()
            )
    equal_state(results["control"], results["resumed"])


@pytest.mark.fast
def test_installed_inert_two_sessions_and_duplicate_request(tmp_path: Path) -> None:
    """Actual native accounting/reaping and CLI grant, with no scientific training."""
    root = tmp_path / "inert"
    script = tmp_path / "inert.sh"
    script.write_text("#!/bin/bash\nexec sleep 60\n")
    prefix = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "chipiron.environments.morpion.bootstrap.operator_cli",
        "start",
        "--work-dir",
        str(root),
    ]
    first = subprocess.run(
        [
            *prefix,
            "--cluster-script",
            str(script),
            "--",
            "--training-device",
            "cpu",
            "--max-elapsed-seconds",
            "1",
            "--min-available-ram-mb",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    state = load_bootstrap_run_state(root / "run_state.json")
    data = state.metadata[managed.KEY]
    scientific = replace(
        state, metadata={k: v for k, v in state.metadata.items() if k != managed.KEY}
    )
    assert (
        len(data["sessions"]) == 1
        and data["active"] is None
        and data["consumed_seconds"] >= 1
    )
    second = subprocess.run(
        [*prefix, "--add-time", "3", "--request-id", "inert-extension"],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    state = load_bootstrap_run_state(root / "run_state.json")
    data = state.metadata[managed.KEY]
    assert (
        len(data["sessions"]) == 2
        and len(data["grants"]) == 1
        and data["authorized_seconds"] == 4
        and data["consumed_seconds"] >= 4
    )
    assert (
        replace(
            state,
            metadata={k: v for k, v in state.metadata.items() if k != managed.KEY},
        )
        == scientific
    )
    before = (root / "run_state.json").read_bytes()
    duplicate = subprocess.run(
        [*prefix, "--add-time", "3", "--request-id", "inert-extension"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (
        duplicate.returncode == 0 and before == (root / "run_state.json").read_bytes()
    )
