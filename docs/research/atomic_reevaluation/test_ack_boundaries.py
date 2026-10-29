"""Supplement existing kill hooks at application and unlink-before-directory-sync."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from tests.environments.morpion_atomic_reevaluation_helpers import (
    accepted,
    publish_proposal,
    setup_atomic,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
)


def child(root: Path, boundary: str) -> None:
    """Reuse PR #79's actual process-kill hooks with the new closed protocol."""
    from tests.environments import test_morpion_checkpoint_patch_receipts as legacy
    from tests.environments.morpion_durable_v2_helpers import scientific_args

    with pytest.MonkeyPatch.context() as adapter:
        adapter.setattr(legacy, "receipt_args", scientific_args)
        adapter.setattr(legacy, "ColdScalarRunner", ClosedPublicationRunner)
        adapter.setattr(
            legacy,
            "run_pipeline_growth_stage",
            lambda args, runner, **_kw: run_closed_publication(args, runner),
        )
        legacy._crash_growth(root, boundary)


@pytest.mark.parametrize(
    "boundary",
    [
        "before_apply",
        "after_apply",
        "after_checkpoint",
        "before_unlink",
        "after_unlink",
    ],
)
def test_application_and_unlink_death(tmp_path: Path, boundary: str) -> None:
    """Coupled cursor survives unlink uncertainty; reappearing exact bytes are safe."""
    args, paths, initial = setup_atomic(tmp_path)
    proposal = publish_proposal(paths)
    raw = paths.pipeline_reevaluation_patch_path.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from docs.research.atomic_reevaluation.test_ack_boundaries import child; child(Path(sys.argv[1]),sys.argv[2])",
            str(tmp_path),
            boundary,
        ],
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 71, result.stderr.decode()
    committed = "unlink" in boundary
    state = load_bootstrap_run_state(paths.run_state_path)
    assert (accepted(state) is not None) == committed
    if not committed:
        assert state == initial
    if boundary == "after_unlink":
        paths.pipeline_reevaluation_patch_path.write_bytes(
            raw
        )  # Simulate unpersisted unlink reappearance, only in this fixture.
    runner = ClosedPublicationRunner()
    final = run_closed_publication(args, runner, max_cycles=0 if committed else 1)
    assert runner.apply_calls == int(not committed)
    assert accepted(final) == proposal.metadata["atomic_reevaluation"]["next_cursor"]
    assert not paths.pipeline_reevaluation_patch_path.exists()


def test_ack_directory_fsync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing ack directory barrier leaves the committed cursor recoverable."""
    args, paths, _ = setup_atomic(tmp_path)
    publish_proposal(paths)
    raw = paths.pipeline_reevaluation_patch_path.read_bytes()
    original = os.fsync

    def fsync(fd: int) -> None:
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        state = load_bootstrap_run_state(paths.run_state_path)
        if (
            path == paths.pipeline_dir
            and state.generation == 2
            and not paths.pipeline_reevaluation_patch_path.exists()
        ):
            message = "injected ack directory failure"
            raise OSError(message)
        original(fd)

    with monkeypatch.context() as injection:
        injection.setattr(os, "fsync", fsync)
        with pytest.raises(OSError, match="ack directory"):
            run_closed_publication(args, ClosedPublicationRunner())
    committed = load_bootstrap_run_state(paths.run_state_path)
    assert accepted(committed)["last_patch_id"] == "proposal-1"
    paths.pipeline_reevaluation_patch_path.write_bytes(raw)
    runner = ClosedPublicationRunner()
    assert run_closed_publication(args, runner, max_cycles=0) == committed
    assert runner.apply_calls == 0
