"""Real local process deaths at immutable Dataset publication and acceptance boundaries."""

import multiprocessing
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import dataset_publication as publication
from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_stage_claim,
    save_pipeline_stage_claim,
)
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
    setup_fenced,
)

PRE = [
    "before_rows",
    "partial_rows",
    "after_rows",
    "before_publication",
    "after_publication",
    "before_acceptance",
    "before_commit",
]
POST = ["after_commit", "before_projection", "after_projection"]


def crashed_worker(args: Any, root: Path, point: str) -> None:
    """Kill the actual stage without running cleanup handlers."""

    def stop() -> None:
        os._exit(81)

    with pytest.MonkeyPatch.context() as patch:
        local_leaderboard(patch, root)
        if point in {"before_rows", "partial_rows", "after_rows"}:
            writer = stages.save_morpion_supervised_rows_streaming

            def write(**kwargs: Any) -> Any:
                if point == "before_rows":
                    stop()
                if point == "partial_rows":
                    original = kwargs["rows"]

                    def partial() -> Iterator[Any]:
                        yield next(iter(original))
                        stop()

                    kwargs["rows"] = partial()
                result = writer(**kwargs)
                stop()
                return result

            patch.setattr(stages, "save_morpion_supervised_rows_streaming", write)
        elif point in {"before_publication", "after_publication"}:
            publish = publication.publish_directory_if_absent

            def publish_hook(*a: Any, **kw: Any) -> None:
                if point == "before_publication":
                    stop()
                publish(*a, **kw)
                stop()

            patch.setattr(publication, "publish_directory_if_absent", publish_hook)
        elif point == "before_acceptance":
            patch.setattr(stages, "accept_dataset", lambda *a: stop())
        elif point in {"before_commit", "after_commit"}:
            save = publication.save_bootstrap_run_state

            def commit(*a: Any, **kw: Any) -> None:
                if point == "before_commit":
                    stop()
                save(*a, **kw)
                stop()

            patch.setattr(publication, "save_bootstrap_run_state", commit)
        else:
            repair = publication._repair

            def projection(*a: Any) -> None:
                if point == "before_projection":
                    stop()
                repair(*a)
                stop()

            patch.setattr(publication, "_repair", projection)
        run_dataset(args)
    os._exit(82)


@pytest.mark.parametrize("point", PRE + POST)
def test_crash_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    """Only the run-state boundary decides whether a visible attempt was accepted."""
    args, paths, _ = setup_fenced(tmp_path)
    args = replace(args, use_backed_up_value=False)
    before = paths.run_state_path.read_bytes()
    process = multiprocessing.get_context("fork").Process(
        target=crashed_worker, args=(args, tmp_path, point)
    )
    process.start()
    process.join(30)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("Dataset crash probe hung")
    assert process.exitcode == 81
    state = load_bootstrap_run_state(paths.run_state_path)
    record = publication.publication_records(state.metadata)[0]
    assert (publication.ACCEPTED in record) == (point in POST)
    if point in PRE:
        assert paths.run_state_path.read_bytes() == before
        claim_path = paths.pipeline_dataset_claim_path_for_generation(1)
        save_pipeline_stage_claim(
            replace(
                load_pipeline_stage_claim(claim_path),
                expires_at_utc="2000-01-01T00:00:00Z",
            ),
            claim_path,
        )
    else:
        before = paths.run_state_path.read_bytes()
    local_leaderboard(monkeypatch, tmp_path)
    completed = run_dataset(args)
    assert completed.dataset_status == "done"
    assert (tmp_path / completed.rows_path).is_file()
    if point in POST:
        assert paths.run_state_path.read_bytes() == before
        assert (
            publication.publication_records(
                load_bootstrap_run_state(paths.run_state_path).metadata
            )[0][publication.ACCEPTED]
            == record[publication.ACCEPTED]
        )


def competing_worker(
    args: Any, root: Path, entered: Any, resume: Any, old: bool
) -> None:
    """Two local processes, with explicit barriers instead of timing assumptions."""
    with pytest.MonkeyPatch.context() as patch:
        local_leaderboard(patch, root)
        if old:
            writer = stages.save_morpion_supervised_rows_streaming

            def pause(**kwargs: Any) -> Any:
                entered.set()
                if not resume.wait(30):
                    os._exit(89)
                return writer(**kwargs)

            patch.setattr(stages, "save_morpion_supervised_rows_streaming", pause)
        try:
            run_dataset(args)
        except ValueError as exc:
            if old and "Conflicting accepted Dataset attempt" in str(exc):
                os._exit(72)
            raise
    os._exit(0)


def test_two_process_workers_and_expired_claim(tmp_path: Path) -> None:
    """A replacement can finish; the still-live old process cannot replace its result."""
    args, paths, _ = setup_fenced(tmp_path)
    args = replace(args, use_backed_up_value=False)
    context = multiprocessing.get_context("fork")
    entered, resume = context.Event(), context.Event()
    old = context.Process(
        target=competing_worker, args=(args, tmp_path, entered, resume, True)
    )
    new = context.Process(
        target=competing_worker, args=(args, tmp_path, entered, resume, False)
    )
    try:
        old.start()
        assert entered.wait(30)
        claim_path = paths.pipeline_dataset_claim_path_for_generation(1)
        save_pipeline_stage_claim(
            replace(
                load_pipeline_stage_claim(claim_path),
                expires_at_utc="2000-01-01T00:00:00Z",
            ),
            claim_path,
        )
        new.start()
        new.join(30)
        assert new.exitcode == 0
        before = paths.run_state_path.read_bytes()
        resume.set()
        old.join(30)
        assert old.exitcode == 72
        assert paths.run_state_path.read_bytes() == before
    finally:
        for process in (old, new):
            if process.pid and process.is_alive():
                process.kill()
                process.join()


@pytest.mark.parametrize(
    "boundary", ["output_directory", "commit_file", "commit_directory", "projection"]
)
def test_fsync_errors_are_not_silent_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """EIO before/after the commit uses observed authority, not an exception heuristic."""
    import errno
    import stat

    from chipiron.environments.morpion.bootstrap import generation_publication_io as io

    args, paths, _ = setup_fenced(tmp_path)
    args = replace(args, use_backed_up_value=False)
    local_leaderboard(monkeypatch, tmp_path)
    before = paths.run_state_path.read_bytes()
    original_sync = os.fsync
    original_commit = publication.save_bootstrap_run_state
    armed = [boundary == "output_directory"]
    fired = []

    def sync(fd: int) -> None:
        name = Path(os.readlink(f"/proc/self/fd/{fd}"))
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        target = (
            (
                boundary == "output_directory"
                and directory
                and name.name == "objects"
                and name.parent.name == publication.NAMESPACE
            )
            or (boundary == "commit_file" and not directory and name.parent == tmp_path)
            or (boundary == "commit_directory" and directory and name == tmp_path)
        )
        if armed[0] and not fired and target:
            fired.append(str(name))
            raise OSError(errno.EIO, "injected fsync failure")
        original_sync(fd)

    def commit(*a: Any, **kw: Any) -> None:
        armed[0] = True
        original_commit(*a, **kw)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(publication, "save_bootstrap_run_state", commit)
    if boundary == "projection":

        def fail(*a: Any) -> None:
            fired.append("projection")
            raise OSError(errno.EIO, "injected projection failure")

        monkeypatch.setattr(publication, "_repair", fail)
    with pytest.raises(OSError):
        run_dataset(args)
    assert fired
    record = io.publication_records(
        load_bootstrap_run_state(paths.run_state_path).metadata
    )[0]
    assert (publication.ACCEPTED in record) == (
        boundary in {"commit_directory", "projection"}
    )
    if boundary in {"output_directory", "commit_file"}:
        assert paths.run_state_path.read_bytes() == before
    else:
        # Restore the repair hook without changing any committed bytes.
        monkeypatch.undo()
        local_leaderboard(monkeypatch, tmp_path)
        accepted = paths.run_state_path.read_bytes()
        assert run_dataset(args).dataset_status == "done"
        assert paths.run_state_path.read_bytes() == accepted
