"""Real process death and filesystem errors with a committed accepted cursor."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from tests.environments import test_morpion_durable_closed_crashes as old
from tests.environments.morpion_atomic_reevaluation_helpers import (
    accepted,
    publish_proposal,
    setup_atomic,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
)


@pytest.mark.parametrize("boundary", old.PRE_COMMIT + old.POST_COMMIT)
def test_atomic_consumer_death(tmp_path: Path, boundary: str) -> None:
    """Reuse every real closed-capsule kill point, including replace-before-fsync."""
    args, paths, initial = setup_atomic(tmp_path)
    proposal = publish_proposal(paths)
    old.run_crash(tmp_path, boundary)
    observed = load_bootstrap_run_state(paths.run_state_path)
    committed = boundary in old.POST_COMMIT
    assert observed.generation == (2 if committed else 1)
    assert (accepted(observed) is not None) == committed
    if not committed:
        assert observed == initial
    runner = ClosedPublicationRunner()
    runner.grow_delta = int(not committed)
    state = run_closed_publication(args, runner, max_cycles=0 if committed else 1)
    assert runner.apply_calls == int(not committed)
    assert accepted(state) == proposal.metadata["atomic_reevaluation"]["next_cursor"]
    assert len(receipt_ledger(state)) == 1
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()
    for _ in range(2):
        runner = ClosedPublicationRunner()
        assert run_closed_publication(args, runner, max_cycles=0) == state
        assert runner.apply_calls == 0


@pytest.mark.parametrize(
    "target",
    [
        "records.jsonl",
        "updates.jsonl",
        "checkpoint",
        "model",
        "capsule_directory",
        "state_directory",
    ],
)
def test_atomic_consumer_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Reuse real fsync fault injections and inspect the coupled committed cursor."""
    monkeypatch.setattr(old, "setup_closed_baseline", setup_atomic)
    monkeypatch.setattr(old, "publish_scientific_patch", publish_proposal)
    old.test_fsync_errors_preserve_observed_commit(tmp_path, monkeypatch, target)
    state = load_bootstrap_run_state(tmp_path / "run_state.json")
    assert accepted(state)["last_patch_id"] == "proposal-1"
    assert len(receipt_ledger(state)) == 1


def producer_child(root: Path, boundary: str) -> None:
    """Terminate the actual producer at the existing create-only visibility boundary."""
    from chipiron.environments.morpion.bootstrap import (
        reevaluation_publication as publication,
    )
    from tests.environments.morpion_atomic_reevaluation_helpers import atomic_worker

    original_link = os.link
    original_sync = os.fsync

    def die() -> None:
        os._exit(73)

    def link(src: Any, dst: Any) -> Any:
        if boundary == "before_link":
            die()
        result = original_link(src, dst)
        if boundary == "after_link":
            die()
        return result

    def sync(fd: int) -> None:
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if boundary == "file_fsync" and path.name.startswith(
            ".reevaluation_patch.json."
        ):
            die()
        original_sync(fd)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(publication.os, "link", link)
        patch.setattr(publication.os, "fsync", sync)
        atomic_worker(root, patch)
    message = "Producer kill boundary not reached"
    raise AssertionError(message)


@pytest.mark.parametrize("boundary", ["file_fsync", "before_link", "after_link"])
def test_producer_death_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """A visible proposal is sufficient; absent publication advances no authority."""
    from tests.environments.morpion_atomic_reevaluation_helpers import atomic_worker

    args, paths, initial = setup_atomic(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from tests.environments.test_morpion_atomic_reevaluation_crashes import producer_child; producer_child(Path(sys.argv[1]),sys.argv[2])",
            str(tmp_path),
            boundary,
        ],
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 73, result.stderr.decode()
    assert load_bootstrap_run_state(paths.run_state_path) == initial
    assert not paths.pipeline_reevaluation_cursor_path.exists()
    if boundary == "after_link":
        assert atomic_worker(tmp_path, monkeypatch).reason == "pending_patch_exists"
    else:
        assert atomic_worker(tmp_path, monkeypatch).patch_written
    state = run_closed_publication(args, ClosedPublicationRunner())
    assert accepted(state)["last_patch_id"] == "worker-proposal"
    assert not paths.pipeline_reevaluation_patch_path.exists()


@pytest.mark.parametrize("fault", ["file_fsync", "directory_fsync", "cleanup"])
def test_producer_io_error_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Neither missing singleton nor post-link EIO can stall new-protocol ack."""
    from chipiron.environments.morpion.bootstrap import (
        reevaluation_publication as publication,
    )
    from tests.environments.morpion_atomic_reevaluation_helpers import atomic_worker

    args, paths, initial = setup_atomic(tmp_path)
    fsync, unlink = os.fsync, Path.unlink

    def sync(fd: int) -> None:
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if (
            fault == "file_fsync" and path.name.startswith(".reevaluation_patch.json.")
        ) or (
            fault == "directory_fsync"
            and path == paths.pipeline_dir
            and paths.pipeline_reevaluation_patch_path.exists()
        ):
            raise OSError(errno.EIO, "injected proposal fsync")
        fsync(fd)

    def cleanup(path: Path, **kwargs: Any) -> None:
        if fault == "cleanup" and path.name.startswith(".reevaluation_patch.json."):
            raise OSError(errno.EIO, "injected proposal cleanup")
        unlink(path, **kwargs)

    with monkeypatch.context() as injection:
        injection.setattr(publication.os, "fsync", sync)
        injection.setattr(Path, "unlink", cleanup)
        with pytest.raises(publication.ReevaluationPatchPublicationError):
            atomic_worker(tmp_path, injection)
    assert load_bootstrap_run_state(paths.run_state_path) == initial
    if fault == "file_fsync":
        assert atomic_worker(tmp_path, monkeypatch).patch_written
    else:
        assert atomic_worker(tmp_path, monkeypatch).reason == "pending_patch_exists"
    state = run_closed_publication(args, ClosedPublicationRunner())
    assert accepted(state)["last_patch_id"] == "worker-proposal"
    assert not paths.pipeline_reevaluation_patch_path.exists()


def competing_publisher(path: Path, proposal: Any, barrier: Any) -> None:
    """Race only two local test processes at create-only publication."""
    from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
        publish_reevaluation_patch_if_absent,
    )

    barrier.wait(timeout=20)
    published = publish_reevaluation_patch_if_absent(proposal, path)
    sys.exit(0 if published else 10)


def test_two_producers_one_accepted_transition(tmp_path: Path) -> None:
    """Concurrent producers capture one predecessor; exactly one can publish."""
    import multiprocessing

    from tests.environments.morpion_atomic_reevaluation_helpers import make_proposal

    args, paths, _ = setup_atomic(tmp_path)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    children = [
        context.Process(
            target=competing_publisher,
            args=(
                paths.pipeline_reevaluation_patch_path,
                make_proposal(paths, f"producer-{i}"),
                barrier,
            ),
        )
        for i in range(2)
    ]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=30)
        assert sorted(child.exitcode for child in children) == [0, 10]
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join()
    runner = ClosedPublicationRunner()
    state = run_closed_publication(args, runner)
    assert runner.apply_calls == 1 and len(receipt_ledger(state)) == 1
    assert accepted(state)["completed_full_pass_count"] == 1
