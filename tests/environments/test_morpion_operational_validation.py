"""Routine experiment checks keep byte validation and exact accepted dependencies."""

import json
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap import (
    dataset_publication,
    training_publication,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    launcher_args_from_cli,
    run_morpion_bootstrap_experiment,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import TempPathFactory


@pytest.fixture(scope="module")
def operational_history(tmp_path_factory: TempPathFactory) -> Path:
    """Generate and train real operational inputs; mutations use independent copies."""
    root = tmp_path_factory.mktemp("operational-validation")
    workspace = root / "workspace"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from docs.research.e2e_bootstrap.worker import initialize; initialize(Path(sys.argv[1]),32,False,'jsonl-zstd-1')",
            str(workspace),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv(
            "MORPION_LEADERBOARD_PATH", str(workspace / "leaderboard.jsonl")
        )
        for stage in ("dataset_worker", "training_worker"):
            run_morpion_bootstrap_experiment(
                launcher_args_from_cli([
                    "--scientific",
                    "--work-dir",
                    str(workspace),
                    "--pipeline-stage",
                    stage,
                ])
            )
    return root


def run_verification(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the same fresh-process validation entrypoint used by experiments."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "docs.research.operational_bootstrap.reproduce",
            str(root),
            "unused",
            "--verify",
            "--validation-level",
            "routine",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_routine_checks_keep_exact_inputs(
    operational_history: Path, tmp_path: Path
) -> None:
    """Routine recovery verifies accepted rows/models without independent re-extraction."""
    shutil.copytree(operational_history / "workspace", tmp_path / "workspace")
    before = (tmp_path / "workspace/run_state.json").read_bytes()
    verified = run_verification(tmp_path)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    result = json.loads((tmp_path / "latest-verification.json").read_text())
    assert result["validation_level"] == "routine-recovery"
    assert result["nodes"] >= 32
    assert result["verified_accepted_identities"][0]["training"]
    assert (tmp_path / "workspace/run_state.json").read_bytes() == before


@pytest.mark.parametrize(
    "damage", ["checkpoint", "science", "model", "rows", "candidate"]
)
def test_routine_checks_refuse_corrupted_required_bytes(
    operational_history: Path, tmp_path: Path, damage: str
) -> None:
    """Changing audit frequency never permits corrupted authoritative or input bytes."""
    workspace = tmp_path / "workspace"
    shutil.copytree(operational_history / "workspace", workspace)
    paths = MorpionBootstrapPaths.from_work_dir(workspace)
    state = load_bootstrap_run_state(paths.run_state_path)
    head = publication_records(state.metadata)[-1]
    if damage == "checkpoint":
        target = workspace / state.latest_runtime_checkpoint_path / "manifest.json"
    elif damage == "science":
        target = workspace / state.latest_tree_snapshot_path
    elif damage == "model":
        target = committed_evaluator(workspace, state)[1] / "param.pt"
    elif damage == "rows":
        proposal = dataset_publication._proposal(paths, head["accepted_dataset"])
        target = workspace / proposal["manifest"]["rows_path"]
    else:
        proposal = training_publication._proposal(paths, head["accepted_training"])
        target = (
            workspace / next(iter(proposal["models"].values()))["path"] / "param.pt"
        )
    target.write_bytes(target.read_bytes() + b"corrupt")
    before = paths.run_state_path.read_bytes()
    verified = run_verification(tmp_path)
    assert verified.returncode != 0
    assert list(tmp_path.glob("validation-failed-*.json"))
    assert paths.run_state_path.read_bytes() == before
    assert not (tmp_path / "latest-verification.json").exists()
