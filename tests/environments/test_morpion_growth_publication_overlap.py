"""Real concurrent publishers and cold recovery across the C7.2 ownership split."""

import json
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.continuous_lifecycle import close_admission
from chipiron.environments.morpion.bootstrap.generation_publication import (
    DurableGenerationPublication,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GrowthOwnershipBlockingIOError,
    artifact_digest,
    committed_evaluator,
    file_digest,
    growth_receipt_ownership,
    publication_namespace,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import activate_model
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.worker_result_io import publication_record
from tests.environments.growth_publication_workload import worker
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_continuous_runtime import fixture
from tests.environments.test_morpion_learning_schedule import monitored as monitored

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def workers(args: Any, root: Path) -> Iterator[tuple[Any, Any, Any]]:
    """Use spawn (no inherited flock), real process groups and bounded test barriers."""
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    children = []

    def start(stage: str) -> Any:
        child = context.Process(
            target=worker, args=(args, root, stage, entered, release)
        )
        child.start()
        children.append(child)
        return child

    try:
        yield start, entered, release
    finally:
        release.set()
        for child in children:
            # Kill the whole session even when its leader has already exited.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.join(timeout=10)
            assert not child.is_alive(), "Test child did not stop"
            child.close()


def finished(child: Any) -> None:
    """Wait once, with a diagnostic bound; never retry a scientific command."""
    child.join(timeout=120)
    assert child.exitcode == 0, "See retained worker log"


@pytest.mark.usefixtures("continuous")
def test_admission_closure_during_search_prevents_stale_growth(tmp_path: Path) -> None:
    """The normal shutdown gate can close during compute and its decision survives."""
    args, paths = fixture(tmp_path / "work", supervisor_limit_seconds=240)
    source = grow(args)
    with workers(args, tmp_path) as (start, entered, release):
        growth = start("growth")
        assert entered.wait(120), "Growth never reached its real search step; see log"
        assert close_admission(paths)
        closed = load_bootstrap_run_state(paths.run_state_path)
        assert not publication.load_frame(paths, closed).state.admission_open
        assert closed.generation == source.generation
        release.set()
        finished(growth)
        assert load_bootstrap_run_state(paths.run_state_path) == closed
        finished(start("cold"))
        assert load_bootstrap_run_state(tmp_path / "c72-cold-state.json") == closed


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("publisher", ["dataset_worker", "training_worker"])
def test_worker_publishes_while_growth_is_inside_real_search(
    tmp_path: Path, publisher: str
) -> None:
    """Acceptance is durable BEFORE release, then both results survive cold restore."""
    args, paths = fixture(tmp_path / "work", supervisor_limit_seconds=240)
    source = grow(args)  # G2 is the first ordinary-cycle learning opportunity.
    generation = source.generation
    if publisher == "training_worker":
        run_dataset(args, generation=generation)
        source = load_bootstrap_run_state(paths.run_state_path)
    evaluator = committed_evaluator(paths.work_dir, source)[0]
    key = "accepted_dataset" if publisher == "dataset_worker" else "accepted_training"
    with workers(args, tmp_path) as (start, entered, release):
        growth = start("growth")
        assert entered.wait(120), "Growth never reached its real search step; see log"
        assert growth.is_alive() and not release.is_set()
        paused = json.loads((tmp_path / "c72-growth-paused.json").read_text())
        assert paused["nodes_after_real_step"] > source.tree_size_at_last_save
        assert load_bootstrap_run_state(paths.run_state_path).generation == generation
        # A second ordinary owner and activation must fail BEFORE touching the
        # live tree, even though shared publication is available to the worker.
        with pytest.raises(GrowthOwnershipBlockingIOError):
            grow(args)
        with pytest.raises(GrowthOwnershipBlockingIOError):
            activate_model(paths, {})
        learning = start(publisher)
        finished(learning)
        accepted = load_bootstrap_run_state(paths.run_state_path)
        assert accepted.generation == generation
        reference = publication_record(accepted, generation)[key]
        assert growth.is_alive() and not release.is_set()
        frame = publication.load_frame(paths, accepted)
        models = {}
        if publisher == "dataset_worker":
            assert frame.state.ready is not None and frame.state.dataset_job is None
        else:
            assert frame.state.last_accepted_round == 1 and frame.state.cohort is None
            proposal = training._proposal(paths, reference)
            for name, item in proposal["models"].items():
                bundle = paths.work_dir / item["path"]
                models[name] = (
                    artifact_digest(bundle),
                    file_digest(bundle / "optimizer.pt"),
                )
            assert set(models) == {"linear", "transformer_v1"}
        # Exercise the real collector while the private attempt is actively pinned.
        with growth_receipt_ownership(paths):
            private = paths.work_dir / publication_namespace(3) / "staging"
            active = list(private.iterdir())
            assert len(active) == 1
            producer = publication_records(accepted.metadata)[-1][
                "scientific_identity"
            ]["producer"]["identity"]
            collector = DurableGenerationPublication(
                paths, scientific_v2_producer=producer, closed_scientific_snapshot=True
            )
            collector.prune_uncommitted(accepted)
            assert active[0].is_dir()
        (tmp_path / "c72-accepted-before-release.json").write_bytes(
            paths.run_state_path.read_bytes()
        )
        release_ns = time.monotonic_ns()
        release.set()
        finished(growth)
        final = load_bootstrap_run_state(paths.run_state_path)
        assert final.generation == generation + 1
        assert publication_record(final, generation)[key] == reference
        assert committed_evaluator(paths.work_dir, final)[0] == evaluator
        current = publication.load_frame(paths, final)
        if publisher == "dataset_worker":
            assert current.state.ready == frame.state.ready
        else:
            assert current.state.last_accepted_round == frame.state.last_accepted_round
            assert current.state.pending_promotion == frame.state.pending_promotion
            for name, item in proposal["models"].items():
                bundle = paths.work_dir / item["path"]
                assert models[name] == (
                    artifact_digest(bundle),
                    file_digest(bundle / "optimizer.pt"),
                )
        finished(start("cold"))
        cold = load_bootstrap_run_state(tmp_path / "c72-cold-state.json")
        assert cold == load_bootstrap_run_state(paths.run_state_path)
        assert cold.generation == generation + 1
        assert publication_record(cold, generation)[key] == reference
        learned = json.loads(
            (tmp_path / ("c72-" + publisher + "-done.json")).read_text()
        )
        grown = json.loads((tmp_path / "c72-growth-done.json").read_text())
        assert paused["at_ns"] < learned["at_ns"] < release_ns < grown["at_ns"]
        (tmp_path / "c72-chronology.json").write_text(
            json.dumps(
                {
                    "publisher": publisher,
                    "source_generation": generation,
                    "committed_generation": cold.generation,
                    "accepted": reference,
                    "checkpoint": cold.latest_runtime_checkpoint_path,
                    "evaluator": evaluator,
                    "ready_before_release": None
                    if frame.state.ready is None
                    else frame.state.ready.to_metadata(),
                    "growth_paused_ns": paused["at_ns"],
                    "worker_durable_ns": learned["at_ns"],
                    "growth_release_ns": release_ns,
                    "growth_committed_ns": grown["at_ns"],
                },
                indent=2,
            )
        )
