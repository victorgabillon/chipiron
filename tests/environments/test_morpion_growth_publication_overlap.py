"""Real concurrent publishers and cold recovery across the C7.2 ownership split."""

import json
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import generation_publication as capsules
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.continuous_lifecycle import (
    close_admission,
    resume_owned,
)
from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
    load_dashboard_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication import (
    DurableGenerationPublication,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    ATTEMPT_MARKER,
    GrowthOwnershipBlockingIOError,
    artifact_digest,
    committed_evaluator,
    file_digest,
    growth_receipt_ownership,
    publication_namespace,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.history import load_bootstrap_history
from chipiron.environments.morpion.bootstrap.model_activation import activate_model
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.worker_result_io import publication_record
from tests.environments.growth_publication_workload import worker
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_continuous_runtime import fixture
from tests.environments.test_morpion_learning_schedule import monitored as monitored

if TYPE_CHECKING:
    from collections.abc import Iterator


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
    assert source.generation == 2
    history_before = paths.history_jsonl_path.read_bytes()
    status_before = paths.latest_status_path.read_bytes()
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
        assert paths.history_jsonl_path.read_bytes() == history_before
        assert paths.latest_status_path.read_bytes() == status_before
        dashboard = load_dashboard_snapshot(paths.work_dir)
        assert dashboard.generation == 2 and dashboard.event["generation"] == 2
        log = (tmp_path / "growth.log").read_text()
        assert "saved=true" not in log
        assert "growth_attempt_discarded" in log and "reason=source_changed" in log
        attempts = [
            json.loads(p.read_text())
            for p in (paths.pipeline_dir / "performance").rglob("growth_attempt-*.json")
        ]
        [discarded] = [a for a in attempts if a["outcome"] == "discarded"]
        assert discarded["source_generation"] == 2
        assert discarded["candidate_generation"] == 3
        assert discarded["elapsed_s"] > 0 and discarded["cpu_s"] >= 0
        finished(start("cold"))
        assert load_bootstrap_run_state(tmp_path / "c72-cold-state.json") == closed
    with growth_receipt_ownership(paths):
        save_bootstrap_run_state(
            resume_owned(paths, closed), paths.run_state_path, durable=True
        )
    retry = tmp_path / "retry"
    retry.mkdir()
    with workers(args, retry) as (start, entered, release):
        growth = start("growth")
        assert entered.wait(120)
        release.set()
        finished(growth)
        accepted = load_bootstrap_run_state(paths.run_state_path)
        assert accepted.generation == 3
        [event] = [
            e
            for e in load_bootstrap_history(paths.history_jsonl_path)
            if e.generation == 3
        ]
        assert (
            event.artifacts.runtime_checkpoint_path
            == accepted.latest_runtime_checkpoint_path
        )
        assert event.artifacts.tree_snapshot_path == accepted.latest_tree_snapshot_path
        assert "/objects/" in event.artifacts.runtime_checkpoint_path
        assert event.metadata["checkpoint_saved"] is True
        assert (retry / "growth.log").read_text().count("saved=true") == 1
        dashboard = load_dashboard_snapshot(paths.work_dir)
        assert dashboard.generation == 3 and dashboard.event["generation"] == 3
        assert (
            dashboard.event["artifacts"]["runtime_checkpoint_path"]
            == accepted.latest_runtime_checkpoint_path
        )
        finished(start("cold"))
        assert load_bootstrap_run_state(retry / "c72-cold-state.json") == accepted
        assert (
            len([
                e
                for e in load_bootstrap_history(paths.history_jsonl_path)
                if e.generation == 3
            ])
            == 1
        )
    (tmp_path / "c72-reporting-acceptance.json").write_text(
        json.dumps({"discarded": discarded, "accepted": dashboard.event}, indent=2)
    )


@pytest.mark.usefixtures("continuous")
def test_accepted_cycle_without_checkpoint_reports_only_accepted_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A committed no-save cycle keeps the existing checkpoint and reports once."""
    args, paths = fixture(tmp_path / "work", supervisor_limit_seconds=240)
    source = grow(args)
    before = load_bootstrap_history(paths.history_jsonl_path)
    checkpoint = source.latest_runtime_checkpoint_path
    monkeypatch.setattr(stages, "should_save_progress", lambda **_: False)
    caplog.clear()
    with caplog.at_level("INFO", logger=stages.__name__):
        accepted = grow(args)
    assert accepted.generation == source.generation == 2
    assert accepted.cycle_index == source.cycle_index + 1
    assert accepted.latest_runtime_checkpoint_path == checkpoint
    history = load_bootstrap_history(paths.history_jsonl_path)
    assert history[:-1] == before
    event = history[-1]
    assert event.generation == 2 and event.cycle_index == accepted.cycle_index
    assert event.artifacts.runtime_checkpoint_path == checkpoint
    assert event.artifacts.tree_snapshot_path == accepted.latest_tree_snapshot_path
    assert event.metadata["cycle_accepted"] is True
    assert event.metadata["checkpoint_saved"] is False
    assert "saved=true" not in caplog.text
    assert caplog.text.count("growth_cycle_done") == 1
    assert "saved=false" in caplog.text
    dashboard = load_dashboard_snapshot(paths.work_dir)
    assert (
        dashboard.generation == 2
        and dashboard.event["cycle_index"] == accepted.cycle_index
    )
    with workers(args, tmp_path) as (start, _entered, _release):
        finished(start("cold"))
    assert load_bootstrap_run_state(tmp_path / "c72-cold-state.json") == accepted


@pytest.mark.usefixtures("continuous")
def test_collector_overlaps_real_staging_to_objects_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real rename races a collector scan while the producer holds its ID pin."""
    args, paths = fixture(tmp_path / "work", supervisor_limit_seconds=240)
    source = grow(args)
    with workers(args, tmp_path) as (start, entered, release):
        growth = start("growth-relocate")
        assert entered.wait(120), "Growth did not reach its publication rename"
        relocation = json.loads((tmp_path / "c72-relocation-before.json").read_text())
        staging, final = Path(relocation["staging"]), Path(relocation["final"])
        assert staging.is_dir() and not final.exists()
        producer = publication_records(source.metadata)[-1]["scientific_identity"][
            "producer"
        ]["identity"]
        collector = DurableGenerationPublication(
            paths, scientific_v2_producer=producer, closed_scientific_snapshot=True
        )
        moved = False
        marker_reads = []
        original_lease, original_read = capsules.capsule_lease, capsules.read_object

        def move_once() -> None:
            nonlocal moved
            if moved:
                return
            entered.clear()
            release.set()
            assert entered.wait(120), "Real capsule rename did not complete"
            assert not release.is_set() and growth.is_alive()
            assert not staging.exists() and final.is_dir()
            moved = True

        @contextmanager
        def lease(work_dir: Path, identifier: str, **kw: Any) -> Iterator[None]:
            if identifier == staging.name and kw.get("exclusive"):
                assert kw["nonblocking"] is True
                move_once()
            with original_lease(work_dir, identifier, **kw):
                yield

        def read_marker(path: Path) -> Any:
            if path == staging / ATTEMPT_MARKER:
                marker_reads.append(str(path))
                # The old collector read this before trying the pin. Make its
                # stale is_file()/read race deterministic using the REAL rename.
                move_once()
            return original_read(path)

        with growth_receipt_ownership(paths):
            # Another genuine private attempt ends without committing its capsule.
            collector.prepare(source)
            try:
                collector.start(source.generation + 1)
                orphan = collector.staging
                assert orphan is not None
            finally:
                collector.close()
            with monkeypatch.context() as patch:
                patch.setattr(capsules, "capsule_lease", lease)
                patch.setattr(capsules, "read_object", read_marker)
                removed = collector.prune_uncommitted(source)
            assert moved and marker_reads == []
            assert removed == [orphan.name] and not orphan.exists()
            assert final.is_dir() and growth.is_alive() and not release.is_set()
            assert load_bootstrap_run_state(paths.run_state_path) == source
        release.set()
        finished(growth)
        accepted = load_bootstrap_run_state(paths.run_state_path)
        assert accepted.generation == 3
        assert publication_records(accepted.metadata)[-1]["id"] == final.name
        with growth_receipt_ownership(paths):
            assert collector.prune_uncommitted(accepted) == []
            collector.validate_state(accepted)
        finished(start("cold"))
        assert load_bootstrap_run_state(tmp_path / "c72-cold-state.json") == accepted
    (tmp_path / "c72-collector-relocation.json").write_text(
        json.dumps(
            {
                "staging": str(staging),
                "active_capsule": str(final),
                "removed_orphan": removed,
                "marker_reads_before_pin": marker_reads,
                "accepted_generation": accepted.generation,
            },
            indent=2,
        )
    )


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
