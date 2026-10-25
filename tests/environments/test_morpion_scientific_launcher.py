"""The operator interface selects the existing complete scientific protocol."""

from dataclasses import replace
from pathlib import Path

import pytest

from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    save_bootstrap_config,
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
from tests.environments.morpion_activation_helpers import setup_activation


def test_scientific_workers_and_explicit_activation(tmp_path: Path) -> None:
    """Actual persisted configuration survives legacy defaults and a cold M2 Growth."""
    _, paths, _ = setup_activation(tmp_path / "work", encoding="jsonl-zstd-1")
    before = load_bootstrap_run_state(paths.run_state_path)
    initial = committed_evaluator(paths.work_dir, before)[0]
    request = tmp_path / "activation.json"

    def run(*options: str) -> None:
        run_morpion_bootstrap_experiment(
            launcher_args_from_cli([
                "--scientific",
                "--work-dir",
                str(paths.work_dir),
                *options,
            ])
        )

    run("--pipeline-stage", "reevaluation", "--reevaluation-max-nodes-per-patch", "2")
    run(
        "--scientific-operation",
        "prepare-activation",
        "--pipeline-generation",
        "1",
        "--activation-request",
        str(request),
    )
    assert (
        committed_evaluator(
            paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
        )[0]
        == initial
    )
    run("--scientific-operation", "activate", "--activation-request", str(request))
    after = load_bootstrap_run_state(paths.run_state_path)
    assert committed_evaluator(paths.work_dir, after)[0] != initial
    committed = paths.run_state_path.read_bytes()
    run("--scientific-operation", "activate", "--activation-request", str(request))
    assert paths.run_state_path.read_bytes() == committed
    run("--pipeline-stage", "growth", "--max-cycles", "1")
    generation = load_bootstrap_run_state(paths.run_state_path).generation
    run("--pipeline-stage", "dataset_worker")
    run("--pipeline-stage", "training_worker", "--skip-evaluator-diagnostics")
    current = load_bootstrap_run_state(paths.run_state_path)
    head = publication_records(current.metadata)[-1]
    assert current.generation == generation > after.generation
    assert "accepted_dataset" in head and "accepted_training" in head
    assert (
        committed_evaluator(paths.work_dir, current)[0]
        == committed_evaluator(paths.work_dir, after)[0]
    )


def test_scientific_initialization_is_explicit_and_preserves_source(
    tmp_path: Path,
) -> None:
    """Fresh initialization uses a real seed and refuses adoption/reinitialization."""
    from tests.environments.morpion_durable_v2_helpers import scientific_args
    from tests.environments.test_morpion_anemone_runner import _make_model_bundle

    seed = tmp_path / "seed"
    _make_model_bundle(seed)
    original = {p.name: p.read_bytes() for p in seed.iterdir() if p.is_file()}
    root = tmp_path / "fresh"
    args = replace(
        scientific_args(root),
        runtime_checkpoint_format="sharded",
        max_growth_steps_per_cycle=2,
        min_visit_count=None,
    )
    config = tmp_path / "config.json"
    save_bootstrap_config(bootstrap_config_from_args(args), config)
    invocation = launcher_args_from_cli([
        "--scientific",
        "--work-dir",
        str(root),
        "--scientific-operation",
        "initialize",
        "--scientific-config",
        str(config),
        "--seed-model",
        str(seed),
        "--initial-nodes",
        "4",
    ])
    with pytest.raises(ValueError, match="explicit initialization"):
        run_morpion_bootstrap_experiment(
            launcher_args_from_cli([
                "--scientific",
                "--work-dir",
                str(root),
                "--pipeline-stage",
                "growth",
            ])
        )
    state = run_morpion_bootstrap_experiment(invocation)
    assert state.generation == 1
    assert original == {p.name: p.read_bytes() for p in seed.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="fresh workspace"):
        run_morpion_bootstrap_experiment(invocation)
