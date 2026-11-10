"""Small real fork contracts before the copy/CLI layer; tracked fresh-process setup."""

import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import managed_sessions as managed
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.execution_profile import (
    bootstrap_execution,
)
from chipiron.environments.morpion.bootstrap.fork_genesis import initialize_fork_genesis
from chipiron.environments.morpion.bootstrap.fork_origin import KEY
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    committed_evaluator,
    copy_artifact,
    file_digest,
    publication_records,
    publish_directory_if_absent,
)
from chipiron.environments.morpion.bootstrap.launcher import _scientific_args
from chipiron.environments.morpion.bootstrap.model_activation import _runner
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.test_morpion_committed_evaluator_restore import fingerprint
from tests.environments.test_morpion_lifecycle_continuation import equal_state


def initialize(root: Path, *, nodes: int = 5000) -> None:
    """Fresh managed parent with a generous native cap; work ends at a barrier."""
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
    from chipiron.environments.morpion.bootstrap.learning_policy import (
        MorpionContinualLearningPolicy,
    )

    policy = MorpionContinualLearningPolicy(
        max_elapsed_seconds=300,
        max_tree_nodes=nodes,
        min_available_ram_mb=1,
        shutdown_grace_seconds=5,
    )
    args = MorpionBootstrapArgs(
        root,
        training_device="cpu",
        execution_profile="strict_math",
        torch_intra_op_threads=1,
        torch_inter_op_threads=1,
        max_growth_steps_per_cycle=2,
        max_rows=128,
        validation_fraction=0.25,
        validation_seed=17,
        continual_learning=policy,
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
    with bootstrap_execution(args):
        prepare_continual_experiment(args, policy, managed_sessions=True)


def semantic(root: Path) -> Any:
    """Cold exact search/RNG/certificate oracle; no scientific publication."""
    from anemone.checkpoints import (
        build_search_checkpoint_payload,
        checkpoint_payload_to_jsonable,
    )
    from anemone.checkpoints.rng_state import rollout_random_generator

    paths = MorpionBootstrapPaths.from_work_dir(root)
    state = load_bootstrap_run_state(paths.run_state_path)
    config = load_bootstrap_config(paths.bootstrap_config_path)
    args = _scientific_args(MorpionBootstrapArgs(root), config)
    with bootstrap_execution(args):
        runner, effective = _runner(paths, state)
        _, bundle = committed_evaluator(root, state)
        try:
            runner.load_or_create(
                root / state.latest_runtime_checkpoint_path, bundle, effective
            )
            runner.validate_model_refresh_certificates()
            rng = rollout_random_generator(runner._runtime)
            return (
                fingerprint(runner),
                checkpoint_payload_to_jsonable(
                    build_search_checkpoint_payload(
                        runner._runtime, state_codec=runner._state_codec
                    )
                ),
                runner._random_generator.getstate(),
                None if rng is None else rng.getstate(),
            )
        finally:
            runner.discard_uncommitted_runtime()


def _invoke_native(root: Path, script: Path) -> None:
    """Real monitor in this fresh numerical process; deterministic supervisor return."""
    from chipiron.environments.morpion.bootstrap.continual_operator import (
        run_monitored_continual_experiment,
    )

    paths = MorpionBootstrapPaths.from_work_dir(root)
    args = _scientific_args(
        MorpionBootstrapArgs(root), load_bootstrap_config(paths.bootstrap_config_path)
    )
    with bootstrap_execution(args):
        run_monitored_continual_experiment(paths, script)


def invoke(root: Path, script: Path) -> None:
    """Exit the actual monitor before another coordinator inspects its close."""
    code = "import sys,torch; from pathlib import Path; torch.set_num_threads(1); torch.set_num_interop_threads(1); from tests.environments.test_morpion_fork_contracts import _invoke_native; _invoke_native(Path(sys.argv[1]),Path(sys.argv[2]))"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(root), str(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def prototype(parent: Path, stage: Path, final: Path) -> None:
    """Exercise only the three missing contracts with verified tiny copied inputs."""
    paths = MorpionBootstrapPaths.from_work_dir(parent)
    state = load_bootstrap_run_state(paths.run_state_path)
    config = load_bootstrap_config(paths.bootstrap_config_path)
    assert managed.inspect(paths, config, state)["active"] is None
    raw = stage / "input" / "checkpoint"
    copy_artifact(parent / state.latest_runtime_checkpoint_path, raw)
    active, bundle = committed_evaluator(parent, state)
    seeds = training.latest_candidate_parents(
        paths, state, list(config.evaluators.evaluators)
    )
    imported = {}
    for name, seed in seeds.items():
        copied = stage / "genesis/fork/seeds" / name
        copy_artifact(parent / seed["model"]["path"], copied)
        imported[name] = {
            "kind": "fork_seed",
            "model": {
                "path": str(copied.relative_to(stage)),
                "sha256": artifact_digest(copied),
            },
            "weights_sha256": file_digest(copied / "param.pt"),
            "origin": {"candidate_name": name, "source_parent": seed},
        }
    marker = managed.capability(final)
    origin = {
        "schema_version": 1,
        "work_dir": str(final),
        "enrollment_id": marker["enrollment_id"],
        "candidate_seeds": imported,
        "dataset_policy": "regenerate_local",
    }
    origin_path = stage / "genesis/fork/origin.json"
    atomic_json(origin_path, origin)
    config = replace(
        config,
        metadata={
            **config.metadata,
            managed.KEY: marker,
            KEY: {
                "path": "genesis/fork/origin.json",
                "sha256": file_digest(origin_path),
            },
        },
    )
    args = _scientific_args(MorpionBootstrapArgs(stage), config)
    with bootstrap_execution(args):
        new = initialize_fork_genesis(
            args,
            config,
            checkpoint=raw,
            incumbent=bundle,
            incumbent_name=active["name"],
        )
    with pytest.raises(ValueError, match=r"capability|root|enrollment"):
        managed.validate(MorpionBootstrapPaths.from_work_dir(stage), config, new)
    assert not any(
        "accepted_training" in r or "accepted_dataset" in r
        for r in publication_records(new.metadata)
    )
    publish_directory_if_absent(stage, final, final.parent)
    assert (
        managed.validate(MorpionBootstrapPaths.from_work_dir(final), config, new)[
            "consumed_seconds"
        ]
        == 0
    )


def run_contract(root: Path) -> None:
    """Real managed closed parent -> new genesis -> real native Adam supersession."""
    parent = root / "parent"
    initialize(parent)
    helper = Path(__file__).with_name("fork_workload.py")
    script = root / "supervisor.sh"
    script.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " -B "
        + shlex.quote(str(helper))
        + ' "$MORPION_WORK_DIR" parent\n'
    )
    invoke(parent, script)
    expected = semantic(parent)
    before = {
        str(p.relative_to(parent)): file_digest(p)
        for p in parent.rglob("*")
        if p.is_file()
    }
    final = root / "child"
    prototype(parent, root / ".stage", final)
    equal_state(expected, semantic(final))
    assert before == {
        str(p.relative_to(parent)): file_digest(p)
        for p in parent.rglob("*")
        if p.is_file()
    }
    script2 = root / "child.sh"
    script2.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " -B "
        + shlex.quote(str(helper))
        + ' "$MORPION_WORK_DIR" child\n'
    )
    invoke(final, script2)
    state = load_bootstrap_run_state(final / "run_state.json")
    records = [r for r in publication_records(state.metadata) if training.ACCEPTED in r]
    assert len(records) == 2
    first = training._proposal(
        MorpionBootstrapPaths.from_work_dir(final), records[0][training.ACCEPTED]
    )
    last = training._proposal(
        MorpionBootstrapPaths.from_work_dir(final), records[-1][training.ACCEPTED]
    )
    assert first["admission"]["schema_version"] == 3
    assert all(p["kind"] == "fork_seed" for p in first["admission"]["parents"].values())
    assert all(
        p.get("kind") != "fork_seed" for p in last["admission"]["parents"].values()
    )
    for model in first["models"].values():
        metadata = json.loads(
            (final / model["path"] / "morpion_manifest.json").read_text()
        )["metadata"]["continuation"]
        assert (
            metadata["optimizer_state_loaded"]
            and metadata["starting_optimizer_step_count"] > 0
        )
    assert before == {
        str(p.relative_to(parent)): file_digest(p)
        for p in parent.rglob("*")
        if p.is_file()
    }


@pytest.mark.exhaustive
def test_three_missing_contracts(tmp_path: Path) -> None:
    """Use the tracked fresh-process numerical pattern, without runtime plugins."""
    code = "import sys,torch; from pathlib import Path; torch.set_num_threads(1); torch.set_num_interop_threads(1); from tests.environments.test_morpion_fork_contracts import run_contract; run_contract(Path(sys.argv[1]))"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
