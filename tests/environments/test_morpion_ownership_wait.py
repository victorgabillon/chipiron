"""Bounded publication waits preserve the existing exclusive ownership fence."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

pytestmark = pytest.mark.fast


def test_learner_waits_for_release_without_touching_authority(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A learner retains its attempt and acquires only after the Growth owner exits."""
    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "2")
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.run_state_path.write_bytes(b"authority stays unchanged")
    waiting, acquired = Event(), Event()

    def learner() -> None:
        waiting.set()
        with growth_receipt_ownership(paths, wait=True):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with growth_receipt_ownership(paths):
            result = executor.submit(learner)
            assert waiting.wait(1)
            assert not acquired.wait(0.1)
            with pytest.raises(BlockingIOError), growth_receipt_ownership(paths):
                pytest.fail("Two Growth owners acquired the same lock")
        result.result(timeout=2)
    assert acquired.is_set()
    assert paths.run_state_path.read_bytes() == b"authority stays unchanged"


def test_learner_wait_timeout_is_bounded(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Timeout refuses the operation; it never steals or expires the owner's lock."""
    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "0.1")
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    with growth_receipt_ownership(paths):
        started = time.monotonic()
        with pytest.raises(BlockingIOError), growth_receipt_ownership(paths, wait=True):
            pytest.fail("A waiter bypassed the owner")
        assert 0.1 <= time.monotonic() - started < 2


@pytest.mark.parametrize("budget", ["nan", "inf", "-1", "601", "invalid"])
def test_invalid_wait_budget_refuses_before_creating_lock(
    tmp_path: Path, monkeypatch: MonkeyPatch, budget: str
) -> None:
    """Malformed launcher settings must not produce an unbounded wait."""
    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", budget)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    with pytest.raises(ValueError), growth_receipt_ownership(paths, wait=True):
        pytest.fail("Invalid ownership budget accepted")
    assert not paths.pipeline_dir.exists()
