"""Real accepted Dataset inputs across producer, Training, recovery and retirement."""

import json
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import prepared_dataset as prepared
from chipiron.environments.morpion.bootstrap import prepared_dataset_storage as storage
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    transformer_v1_morpion_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.prepared_dataset_recipe import (
    dataset_recipe,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_candidate_execution import assert_state_equal
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_learning_schedule import monitored as monitored


@pytest.fixture
def enabled(monitored: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real local monitored identity and the public existing opt-in."""
    monkeypatch.setenv("CHIPIRON_SHARED_PREPARATION", "1")
    monkeypatch.delenv("CHIPIRON_SHARED_PREPARATION_REUSE", raising=False)
    monkeypatch.delenv("CHIPIRON_STATE_SPLIT_CACHE_DIR", raising=False)


def fixture(root: Path, *, training_cadence: int = 1) -> tuple[Any, Any, int]:
    """Create a small legal real source and qualified flat/relational cohort."""
    family = MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec("linear", "linear", None, 1, 2, 0.001),
            "transformer_v1": replace(
                transformer_v1_morpion_evaluator_spec(),
                num_epochs=1,
                batch_size=2,
                entity_d_model=8,
                entity_n_head=2,
                entity_n_layer=1,
                entity_dim_feedforward=8,
            ),
        }
    )
    args, paths = setup_continual(
        root, memory_floor_mb=1, evaluators=family, training_cadence=training_cadence
    )
    source = grow(args)
    return args, paths, source.generation


def one(paths: Any) -> tuple[str, dict[str, Any]]:
    """Read the sole operational intent from authoritative run state."""
    data = storage.entries(load_bootstrap_run_state(paths.run_state_path))
    assert len(data) == 1
    return next(iter(data.items()))


def child(
    args: Any, generation: int, action: str, *, point: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run a real separate owned process/session; boundedly reap even on assertion."""
    path = Path(args.work_dir) / f"fixture-{action}-{point}.pickle"
    path.write_bytes(pickle.dumps((args, generation, action, point)))
    command = [
        sys.executable,
        "-m",
        "tests.environments.prepared_dataset_workload",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=90)
        return subprocess.CompletedProcess(command, process.returncode, output)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.fast
@pytest.mark.usefixtures("enabled")
def test_real_worker_handoff_and_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepare in Dataset, consume actual Training without a second copy/build, replay retired."""
    args, paths, generation = fixture(tmp_path / "run")
    assert not torch.cuda.is_initialized()
    rng = torch.get_rng_state().clone()
    run_dataset(args, generation=generation)
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    key, record = one(paths)
    assert record["phase"] == "COMPLETE"
    value = storage.ready_descriptor(paths, record)
    assert (
        storage.verify_reference(paths, record["rows"]).read_bytes()
        == storage.verify_reference(paths, value["rows"]).read_bytes()
    )
    ready_before = (storage.directory(paths, key) / "ready.json").read_bytes()
    outcome = child(args, generation, "consume")
    assert outcome.returncode == 0, outcome.stdout
    assert (storage.directory(paths, key) / "ready.json").read_bytes() == ready_before
    assert prepared.retire(paths, key)
    root = storage.directory(paths, key)
    assert not (root / "rows.jsonl").exists()
    assert (root / ".lease").is_file() and (root / "retired.json").is_file()
    assert (root / ".rows.jsonl.shared-preparation-v1/entry/split.json").is_file()
    assert not storage.entries(load_bootstrap_run_state(paths.run_state_path))
    # Accepted retries bypass all operational preparation, even after its retirement.
    monkeypatch.setattr(
        training,
        "_compare_incumbent",
        lambda *a, **k: pytest.fail("accepted retry recomputed"),
    )
    result = run_pipeline_training_stage(args, generation=generation)
    assert result.training_status == "done"


@pytest.mark.usefixtures("enabled")
@pytest.mark.parametrize("point", ["copy", "native", "ready"])
def test_real_producer_death_restart(tmp_path: Path, point: str) -> None:
    """Death at each durable boundary keeps scientific acceptance and original deadline."""
    args, paths, generation = fixture(tmp_path / "run")
    failed = child(args, generation, "produce", point=point)
    assert failed.returncode == 73, failed.stdout
    key, record = one(paths)
    original = record["deadline"], record["utc_deadline"], record["accepted"]
    accepted = publication_records(
        load_bootstrap_run_state(paths.run_state_path).metadata
    )[-1]["accepted_dataset"]
    assert accepted == record["accepted"]
    restarted = child(args, generation, "produce")
    assert restarted.returncode == 0, restarted.stdout
    _, after = one(paths)
    assert after["phase"] == "COMPLETE"
    assert (after["deadline"], after["utc_deadline"], after["accepted"]) == original
    storage.ready_descriptor(paths, after)
    assert storage.directory(paths, key).name == accepted["descriptor_sha256"]


@pytest.mark.usefixtures("enabled")
@pytest.mark.parametrize(
    "damage", ["rows", "recipe", "shape", "order", "hash", "symlink"]
)
def test_preconsumer_integrity(tmp_path: Path, damage: str) -> None:
    """Refuse invalid content without changing accepted rows or replacing current inputs."""
    args, paths, generation = fixture(tmp_path / "run")
    run_dataset(args, generation=generation)
    key, record = one(paths)
    original = storage.verify_reference(paths, record["rows"]).read_bytes()
    root = storage.directory(paths, key)
    target = root / ".rows.jsonl.shared-preparation-v1/entry/flat.pt"
    if damage == "rows":
        (root / "rows.jsonl").write_bytes(b"changed")
    elif damage == "recipe":
        with growth_receipt_ownership(paths):
            state = load_bootstrap_run_state(paths.run_state_path)
            data = storage.entries(state)
            data[key] = {**record, "recipe": {**record["recipe"], "seed": 99}}
            save_bootstrap_run_state(
                storage.updated(state, data), paths.run_state_path, durable=True
            )
    elif damage == "order":
        rows = (root / "rows.jsonl").read_bytes().splitlines(keepends=True)
        (root / "rows.jsonl").write_bytes(b"".join(reversed(rows)))
    elif damage == "symlink":
        target.unlink()
        target.symlink_to(storage.verify_reference(paths, record["rows"]))
    elif damage == "shape":
        torch.save({"features": torch.zeros(1, 1)}, target)
    else:
        target.write_bytes(b"bad digest")
    if damage == "recipe":
        # Incompatible pre-consumption recipes deliberately keep the old private path.
        result = run_pipeline_training_stage(args, generation=generation)
        assert result.training_status == "done"
    else:
        with pytest.raises((ValueError, OSError)):
            storage.ready_descriptor(paths, one(paths)[1])
    assert storage.verify_reference(paths, record["rows"]).read_bytes() == original


@pytest.mark.usefixtures("enabled")
def test_owned_consumer_corruption_and_retirement_refusal(tmp_path: Path) -> None:
    """Keep the native lease throughout readers and fail after consumed-input corruption."""
    args, paths, generation = fixture(tmp_path / "run")
    run_dataset(args, generation=generation)
    key, record = one(paths)
    from chipiron.environments.morpion.bootstrap.control import load_bootstrap_control
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_manifest,
    )

    manifest = load_pipeline_manifest(
        paths.pipeline_manifest_path_for_generation(generation)
    )
    config = training.configuration(
        args,
        args.resolved_evaluators_config(),
        load_bootstrap_control(paths.control_path),
    )
    with training.admit_training(
        paths, manifest, config, ttl_seconds=60, owner=None
    ) as attempt:
        assert isinstance(attempt, training.TrainingAttempt)
        with (
            pytest.raises(ValueError, match=r"changed|checksum"),
            prepared.borrow(paths, args, attempt.admission) as borrowed,
        ):
            assert borrowed is not None
            assert not prepared.retire(paths, key)
            borrowed[0].write_bytes(b"corrupt after consumption")
    assert storage.verify_reference(paths, record["rows"]).is_file()


@pytest.mark.usefixtures("enabled")
def test_default_off_unsupported_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep old paths and make resource refusal leave accepted history intact."""
    args, paths, generation = fixture(tmp_path / "run")
    monkeypatch.setenv("CHIPIRON_SHARED_PREPARATION", "0")
    assert dataset_recipe(args, paths.work_dir / "rows.jsonl") is None
    run_dataset(args, generation=generation)
    assert storage.KEY not in load_bootstrap_run_state(paths.run_state_path).metadata
    monkeypatch.setenv("CHIPIRON_SHARED_PREPARATION", "1")
    unsupported = replace(
        args,
        evaluators_config=MorpionEvaluatorsConfig(
            evaluators={
                "linear": args.resolved_evaluators_config().evaluators["linear"]
            }
        ),
    )
    assert dataset_recipe(unsupported, paths.work_dir / "rows.jsonl") is None
    args2, paths2, generation2 = fixture(tmp_path / "other")
    monkeypatch.setenv("CHIPIRON_SHARED_PREPARATION_MAX_SCRATCH_MB", "1")
    run_dataset(args2, generation=generation2)
    assert not storage.entries(load_bootstrap_run_state(paths2.run_state_path))
    assert (
        "accepted_dataset"
        in publication_records(
            load_bootstrap_run_state(paths2.run_state_path).metadata
        )[-1]
    )


@pytest.mark.usefixtures("enabled")
def test_two_round_exact_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare real weights/Adam/RNG/selection with private-copy reference across own parents."""
    outputs: dict[str, list[Any]] = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("CHIPIRON_SHARED_PREPARATION", mode)
        args, paths, generation = fixture(tmp_path / mode / "run")
        outputs[mode] = []
        for number in range(2):
            if number:
                generation = grow(args).generation
            run_dataset(args, generation=generation)
            result = run_pipeline_training_stage(args, generation=generation)
            state = load_bootstrap_run_state(paths.run_state_path)
            accepted = publication_records(state.metadata)[-1]["accepted_training"]
            proposal = training._proposal(paths, accepted)
            outputs[mode].append((paths, result, proposal))
            prepared.reconcile(paths, args)
    for left, right in zip(outputs["0"], outputs["1"], strict=True):
        assert left[1].selected_evaluator_name == right[1].selected_evaluator_name
        for name in left[2]["models"]:
            metadata = []
            for paths, _, proposal in (left, right):
                model = paths.work_dir / proposal["models"][name]["path"]
                metadata.append(
                    json.loads((model / "morpion_manifest.json").read_text())[
                        "metadata"
                    ]
                )
            for artifact in ("param.pt", "optimizer.pt"):
                values = [
                    torch.load(
                        paths.work_dir / proposal["models"][name]["path"] / artifact,
                        weights_only=True,
                    )
                    for paths, _, proposal in (left, right)
                ]
                assert_state_equal(*values)
            for field in ("train_loss", "validation_loss", "state_split"):
                assert metadata[0][field] == metadata[1][field]
            for field in (
                "rng_before_initialization",
                "rng_before_training",
                "rng_after_training",
                "optimizer_steps",
                "cumulative_optimizer_steps",
                "parent_optimizer_steps",
            ):
                assert (
                    metadata[0]["continuation"][field]
                    == metadata[1]["continuation"][field]
                )
