"""Independent synthetic producers and consumers, with deterministic barriers."""

from __future__ import annotations

import json
import select
import subprocess
import sys
from pathlib import Path

import pytest

from chipiron.environments.morpion.bootstrap import (
    load_bootstrap_run_state,
    run_pipeline_growth_stage,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    observe_pending_patch,
    receipt_ledger,
)
from tests.environments.morpion_patch_receipt_helpers import (
    PATCHED,
    ColdScalarRunner,
    publish_fixture_patch,
    setup_receipt_baseline,
)


def _line(process: subprocess.Popen[str]) -> str:
    """Bound every cross-process barrier; a broken child cannot strand CI."""
    assert process.stdout is not None
    assert select.select([process.stdout], [], [], 15)[0], "Child barrier timed out"
    return process.stdout.readline().strip()


def test_growth_ownership_is_not_stolen_and_death_releases_lock(tmp_path: Path) -> None:
    """A kernel lease covers restore through ack, independently of JSON claim TTLs."""
    args, paths = setup_receipt_baseline(tmp_path)
    publish_fixture_patch(paths)
    script = """
import sys
from pathlib import Path
from chipiron.environments.morpion.bootstrap.bootstrap_paths import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import growth_receipt_ownership
with growth_receipt_ownership(MorpionBootstrapPaths.from_work_dir(Path(sys.argv[1]))):
    print('locked', flush=True)
    sys.stdin.readline()
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as owner:
        try:
            assert _line(owner) == "locked"
            runner = ColdScalarRunner()
            with pytest.raises(BlockingIOError):
                run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
            assert (
                runner.apply_calls == 0
                and paths.pipeline_reevaluation_patch_path.exists()
            )
        finally:
            owner.kill()
            owner.wait(timeout=10)
    runner = ColdScalarRunner()
    state = run_pipeline_growth_stage(args, runner, checkpoint_bound_receipts=True)
    assert runner.apply_calls == 1 and state.generation == 2
    assert not paths.pipeline_reevaluation_patch_path.exists()


@pytest.mark.parametrize("publication_phase", ["durable", "after_link"])
def test_producer_cursor_barrier_and_new_patch_after_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publication_phase: str
) -> None:
    """A delayed producer cursor fences ack; a subsequent publication survives old ack."""
    args, paths = setup_receipt_baseline(tmp_path)
    script = """
import os, sys
from pathlib import Path
from chipiron.environments.morpion.bootstrap.bootstrap_paths import MorpionBootstrapPaths
from tests.environments.morpion_patch_receipt_helpers import fixture_patch, publish_fixture_patch, save_fixture_cursor
paths = MorpionBootstrapPaths.from_work_dir(Path(sys.argv[1]))
phase = sys.argv[2]
original_link = os.link
first_link = True
def link(source, destination):
    global first_link
    original_link(source, destination)
    if first_link and phase == 'after_link':
        first_link = False
        print('published', flush=True)
        assert sys.stdin.readline().strip() == 'durable'
os.link = link
patch = publish_fixture_patch(paths, cursor=False)
print('durable' if phase == 'after_link' else 'published', flush=True)
assert sys.stdin.readline().strip() == 'cursor'
save_fixture_cursor(paths, patch)
print('cursor', flush=True)
assert sys.stdin.readline().strip() == 'next'
publish_fixture_patch(paths, fixture_patch('patch-2'))
print('next', flush=True)
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), publication_phase],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as producer:
        try:
            assert _line(producer) == "published"
            runner = ColdScalarRunner()
            state = run_pipeline_growth_stage(
                args, runner, checkpoint_bound_receipts=True
            )
            assert runner.apply_calls == 1 and state.generation == 2
            assert paths.pipeline_reevaluation_patch_path.exists()
            assert producer.stdin is not None
            if publication_phase == "after_link":
                producer.stdin.write("durable\n")
                producer.stdin.flush()
                assert _line(producer) == "durable"
            producer.stdin.write("cursor\n")
            producer.stdin.flush()
            assert _line(producer) == "cursor"
            original_unlink = Path.unlink

            def unlink_and_publish(path: Path, missing_ok: bool = False) -> None:
                original_unlink(path, missing_ok=missing_ok)
                if path == paths.pipeline_reevaluation_patch_path:
                    assert producer.stdin is not None
                    producer.stdin.write("next\n")
                    producer.stdin.flush()
                    assert _line(producer) == "next"

            with monkeypatch.context() as barrier:
                barrier.setattr(Path, "unlink", unlink_and_publish)
                restarted = ColdScalarRunner()
                run_pipeline_growth_stage(
                    args, restarted, max_cycles=0, checkpoint_bound_receipts=True
                )
            assert restarted.apply_calls == 0
            pending = observe_pending_patch(paths.pipeline_reevaluation_patch_path)
            assert pending is not None and pending.patch.patch_id == "patch-2"
            state = load_bootstrap_run_state(paths.run_state_path)
            assert set(receipt_ledger(state)) == {"patch-1"}
            assert state.latest_runtime_checkpoint_path is not None
            assert (
                json.loads(
                    (tmp_path / state.latest_runtime_checkpoint_path).read_bytes()
                )
                == PATCHED
            )
            assert producer.wait(timeout=10) == 0
        finally:
            if producer.poll() is None:
                producer.kill()
                producer.wait(timeout=10)
