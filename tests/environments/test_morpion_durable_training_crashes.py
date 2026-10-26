"""Local process deaths and live claim takeover at the Training commit boundary."""

import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import training_publication as publication
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_stage_claim,
    save_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.worker_result_io import publication_record
from tests.environments.morpion_durable_training_helpers import setup_training
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard

PRE = [
    "before_training",
    "during_training",
    "serialization",
    "before_publication",
    "after_publication",
    "publication_sync",
    "before_acceptance",
    "before_replace",
]
POST = [
    "after_replace",
    "after_acceptance",
    "before_projection",
    "after_projection",
    "acknowledgement",
]


def crashed_worker(args: Any, point: str) -> None:
    """Exit without cleanup from the actual CPU trainer and publication path."""
    from chipiron.environments.morpion.bootstrap import cycle_training, pipeline_claims
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        service,
    )

    def stop(*_a: Any, **_kw: Any) -> Any:
        os._exit(81)

    with pytest.MonkeyPatch.context() as patch:
        if point == "before_training":
            patch.setattr(cycle_training, "train_and_select_evaluators_streaming", stop)
        elif point == "during_training":
            step = torch.optim.Adam.step

            def midway(*a: Any, **kw: Any) -> Any:
                step(*a, **kw)
                stop()

            patch.setattr(torch.optim.Adam, "step", midway)
        elif point == "serialization":

            def partial(_model: Any, output: Any, **_kw: Any) -> Any:
                path = Path(output)
                path.mkdir(parents=True)
                (path / "param.pt").write_bytes(b"partial weight serialization")
                stop()

            patch.setattr(service, "save_morpion_model_bundle", partial)
        elif point in {"before_publication", "after_publication"}:
            publish = publication.publish_directory_if_absent

            def hook(*a: Any, **kw: Any) -> Any:
                if point == "before_publication":
                    stop()
                publish(*a, **kw)
                stop()

            patch.setattr(publication, "publish_directory_if_absent", hook)
        elif point == "publication_sync":
            from chipiron.environments.morpion.bootstrap import (
                generation_publication_io as io,
            )

            sync = io.sync_directory

            def sync_hook(path: Path) -> None:
                if path.name == "objects" and path.parent.name == publication.NAMESPACE:
                    stop()
                sync(path)

            patch.setattr(io, "sync_directory", sync_hook)
        elif point == "before_acceptance":
            patch.setattr(publication, "accept_training", stop)
        elif point in {"before_replace", "after_replace"}:
            rename = os.replace

            def replace_hook(src: Any, dst: Any) -> None:
                if Path(dst).name == "run_state.json" and point == "before_replace":
                    stop()
                rename(src, dst)
                if Path(dst).name == "run_state.json":
                    stop()

            patch.setattr(os, "replace", replace_hook)
        elif point == "after_acceptance":
            save = publication.save_bootstrap_run_state

            def commit(*a: Any, **kw: Any) -> None:
                save(*a, **kw)
                stop()

            patch.setattr(publication, "save_bootstrap_run_state", commit)
        elif point == "acknowledgement":
            patch.setattr(pipeline_claims, "release_pipeline_stage_claim", stop)
        else:
            repair = publication._repair

            def repair_hook(*a: Any) -> None:
                if point == "before_projection":
                    stop()
                repair(*a)
                stop()

            patch.setattr(publication, "_repair", repair_hook)
        run_pipeline_training_stage(args, generation=1)
    os._exit(82)


@pytest.mark.parametrize(
    "point",
    [
        pytest.param(
            value,
            marks=pytest.mark.fast
            if value in ["after_replace", "before_replace"]
            else pytest.mark.exhaustive,
        )
        for value in PRE + POST
    ],
)
def test_actual_crash_recovery(tmp_path: Path, monkeypatch: Any, point: str) -> None:
    """Only the observed authoritative replacement determines acceptance on restart."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    before = paths.run_state_path.read_bytes()
    active = paths.pipeline_active_model_path.read_bytes()
    process = multiprocessing.get_context("spawn").Process(
        target=crashed_worker, args=(args, point)
    )
    process.start()
    process.join(50)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("Local Training crash probe hung")
    assert process.exitcode == 81
    state = load_bootstrap_run_state(paths.run_state_path)
    record = publication_record(state, 1)
    assert (publication.ACCEPTED in record) == (point in POST)
    if point in PRE:
        assert paths.run_state_path.read_bytes() == before
        claim = paths.pipeline_training_claim_path_for_generation(1)
        save_pipeline_stage_claim(
            replace(
                load_pipeline_stage_claim(claim), expires_at_utc="2000-01-01T00:00:00Z"
            ),
            claim,
        )
        assert run_pipeline_training_stage(args, generation=1).training_status == "done"
    else:
        reference = record[publication.ACCEPTED]
        root = publication._directory(paths, reference["attempt_id"])
        exact = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        assert run_pipeline_training_stage(args, generation=1).training_status == "done"
        assert {p: p.read_bytes() for p in exact} == exact
        assert (
            publication_record(load_bootstrap_run_state(paths.run_state_path), 1)[
                publication.ACCEPTED
            ]
            == reference
        )
    assert paths.pipeline_active_model_path.read_bytes() == active
    assert not paths.pipeline_training_cursor_path.exists()


def racing_worker(args: Any, ready: Any, resume: Any) -> None:
    """Hold an old live attempt until the replacement has accepted."""
    accept = publication.accept_training

    def paused(paths: Any, reference: Any) -> Any:
        ready.set()
        if not resume.wait(40):
            os._exit(90)
        return accept(paths, reference)

    publication.accept_training = paused
    try:
        run_pipeline_training_stage(args, generation=1)
    except ValueError:
        os._exit(72)
    os._exit(73)


def test_two_live_processes(tmp_path: Path, monkeypatch: Any) -> None:
    """A still-running expired worker cannot overwrite the replacement winner."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    ctx = multiprocessing.get_context("spawn")
    ready, resume = ctx.Event(), ctx.Event()
    old = ctx.Process(target=racing_worker, args=(args, ready, resume))
    old.start()
    try:
        assert ready.wait(40)
        claim = paths.pipeline_training_claim_path_for_generation(1)
        save_pipeline_stage_claim(
            replace(
                load_pipeline_stage_claim(claim), expires_at_utc="2000-01-01T00:00:00Z"
            ),
            claim,
        )
        run_pipeline_training_stage(args, generation=1)
        accepted = paths.run_state_path.read_bytes()
        resume.set()
        old.join(30)
        assert old.exitcode == 72
        assert paths.run_state_path.read_bytes() == accepted
    finally:
        if old.is_alive():
            old.kill()
            old.join()


@pytest.mark.parametrize("fault", ["file_sync", "directory_sync", "projection"])
def test_commit_faults(tmp_path: Path, monkeypatch: Any, fault: str) -> None:
    """Report failed persistence without turning post-replace acceptance into failure."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    commit = publication.save_bootstrap_run_state

    def failing_commit(*a: Any, **kw: Any) -> Any:
        sync = os.fsync
        count = 0

        def fail(fd: Any) -> None:
            nonlocal count
            count += 1
            if count == (1 if fault == "file_sync" else 2):
                raise OSError("injected commit fsync")  # noqa: TRY003
            sync(fd)

        with pytest.MonkeyPatch.context() as local:
            local.setattr(os, "fsync", fail)
            return commit(*a, **kw)

    if fault == "projection":

        def fail_projection(*a: Any) -> None:
            raise OSError("injected projection")  # noqa: TRY003

        monkeypatch.setattr(publication, "_repair", fail_projection)
    else:
        monkeypatch.setattr(publication, "save_bootstrap_run_state", failing_commit)
    with pytest.raises(OSError, match="injected"):
        run_pipeline_training_stage(args, generation=1)
    accepted = publication.ACCEPTED in publication_record(
        load_bootstrap_run_state(paths.run_state_path), 1
    )
    assert accepted == (fault != "file_sync")
