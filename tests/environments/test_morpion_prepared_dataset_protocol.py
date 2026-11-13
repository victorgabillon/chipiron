"""Finite owned boundary/race tests for Dataset preparation, not a bootstrap experiment."""

import os
import pickle
import select
import subprocess
import sys
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from chipiron.environments.morpion.bootstrap import prepared_dataset as prepared
from chipiron.environments.morpion.bootstrap import prepared_dataset_storage as storage
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_stage_claim,
    save_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow
from tests.environments.test_morpion_learning_schedule import monitored as monitored
from tests.environments.test_morpion_prepared_dataset import child, fixture, one
from tests.environments.test_morpion_prepared_dataset import enabled as enabled


@pytest.mark.usefixtures("enabled")
def test_different_root_overlap_and_same_root_exclusion(tmp_path: Path) -> None:
    """Hold a real native consumer while another accepted Dataset prepares, using pipes."""
    # Immutable cadence permits two committed sources before the historical barrier.
    # Explicit stage admissions test reuse, not a new automatic scheduling policy.
    args, paths, first = fixture(tmp_path / "run", training_cadence=3)
    run_dataset(args, generation=first)
    key, _ = one(paths)
    second = grow(args).generation
    assert second > first
    config = paths.work_dir / "hold.pickle"
    config.write_bytes(pickle.dumps((args, first, "hold", "")))
    ready_r, ready_w = os.pipe()
    release_r, release_w = os.pipe()
    env = {**os.environ, "J41_READY_FD": str(ready_w), "J41_RELEASE_FD": str(release_r)}
    log = paths.work_dir / "held.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.environments.prepared_dataset_workload",
                str(config),
            ],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            pass_fds=(ready_w, release_r),
            start_new_session=True,
        )
        os.close(ready_w)
        os.close(release_r)
        try:
            assert select.select([ready_r], [], [], 60)[0], log.read_text()
            assert os.read(ready_r, 1) == b"R", log.read_text()
            assert not prepared.retire(paths, key)
            with pytest.raises(BlockingIOError), storage.lease(paths, key):
                pass
            # A real held consumer owns no Growth lock.
            with growth_receipt_ownership(paths):
                pass
            run_dataset(args, generation=second)
            data = storage.entries(load_bootstrap_run_state(paths.run_state_path))
            assert len(data) == 2 and data[key]["phase"] == "CONSUMING"
            assert any(
                r["generation"] == second and r["phase"] == "COMPLETE"
                for r in data.values()
            )
            assert (
                sum(r["reserved_bytes"] for r in data.values())
                <= args.continual_learning.max_scratch_mb * 2**20
            )
            os.write(release_w, b"G")
            assert process.wait(timeout=30) == 0, log.read_text()
        finally:
            os.close(ready_r)
            os.close(release_w)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    assert prepared.retire(paths, key)  # Unconsumed, now quiescent operational inputs.


@pytest.mark.usefixtures("enabled")
def test_parent_advances_after_dataset_preparation(tmp_path: Path) -> None:
    """Preparation never captures parents: later admission uses each newly accepted lineage."""
    # Immutable cadence permits two committed sources before the historical barrier.
    # Explicit stage admissions test reuse, not a new automatic scheduling policy.
    args, paths, first = fixture(tmp_path / "run", training_cadence=3)
    run_dataset(args, generation=first)
    second = grow(args).generation
    assert second > first
    run_dataset(args, generation=second)
    data = storage.entries(load_bootstrap_run_state(paths.run_state_path))
    assert len(data) == 2 and all(r["phase"] == "COMPLETE" for r in data.values())
    run_pipeline_training_stage(args, generation=first)
    state = load_bootstrap_run_state(paths.run_state_path)
    accepted = next(
        r["accepted_training"]
        for r in publication_records(state.metadata)
        if r["generation"] == first
    )
    run_pipeline_training_stage(args, generation=second)
    state = load_bootstrap_run_state(paths.run_state_path)
    reference = next(
        r["accepted_training"]
        for r in publication_records(state.metadata)
        if r["generation"] == second
    )
    proposal = training._proposal(paths, reference)
    assert all(
        p["accepted_training"] == accepted
        for p in proposal["admission"]["parents"].values()
    )


@pytest.mark.usefixtures("enabled")
def test_claim_replacement_fences_recovery(tmp_path: Path) -> None:
    """A late producer cannot complete after replacement, even with accepted rows."""
    args, paths, generation = fixture(tmp_path / "run")
    failed = child(args, generation, "produce", point="copy")
    assert failed.returncode == 73, failed.stdout
    _, record = one(paths)
    path = paths.pipeline_dataset_claim_path_for_generation(generation)
    claim = load_pipeline_stage_claim(path)
    save_pipeline_stage_claim(replace(claim, claim_id=uuid.uuid4().hex), path)
    result = child(args, generation, "produce")
    assert result.returncode != 0 and "claim replaced" in result.stdout, result.stdout
    assert one(paths)[1]["accepted"] == record["accepted"]


@pytest.mark.usefixtures("enabled")
def test_lifecycle_suspends_without_budget_reset_and_refuses_fork(
    tmp_path: Path,
) -> None:
    """Use the exact managed-close/fork hooks on an unfinished, positively dead producer."""
    args, paths, generation = fixture(tmp_path / "run")
    result = child(args, generation, "produce", point="copy")
    assert result.returncode == 73, result.stdout
    key, before = one(paths)
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        state = storage.settle_owned(paths, state)
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    after = one(paths)[1]
    assert after["phase"] == "SUSPENDED"
    assert after["deadline"] == before["deadline"] and after["owner"] == before["owner"]
    with pytest.raises(ValueError, match="Unsettled"):
        storage.require_settled(paths, load_bootstrap_run_state(paths.run_state_path))
    result = child(args, generation, "produce")
    assert result.returncode == 0, result.stdout
    storage.require_settled(paths, load_bootstrap_run_state(paths.run_state_path))
    assert prepared.retire(paths, key)


@pytest.mark.usefixtures("enabled")
def test_expired_preparation_never_renews(tmp_path: Path) -> None:
    """Corrupt/expired clocks refuse recovery before copying or rebuilding anything."""
    args, paths, generation = fixture(tmp_path / "run")
    result = child(args, generation, "produce", point="copy")
    assert result.returncode == 73, result.stdout
    key, record = one(paths)
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        record = {**record, "deadline": 1.0, "utc_deadline": 1.0}
        save_bootstrap_run_state(
            storage.updated(state, {key: record}), paths.run_state_path, durable=True
        )
    before = paths.run_state_path.read_bytes()
    with pytest.raises(ValueError, match=r"exhausted|clock"):
        prepared.prepare_accepted(paths, args, generation)
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.usefixtures("enabled")
def test_training_waits_for_live_preparation(tmp_path: Path) -> None:
    """Pause a real producer after acceptance; admitted Training borrows its eventual output."""
    args, paths, generation = fixture(tmp_path / "run")
    ready_r, ready_w = os.pipe()
    release_r, release_w = os.pipe()
    consumer_r, consumer_w = os.pipe()
    children = []
    with (paths.work_dir / "race.log").open("w") as output:
        try:
            for action, point in (
                ("produce", "hold-copy"),
                ("consume", "announce-admission"),
            ):
                config = paths.work_dir / f"race-{action}.pickle"
                config.write_bytes(pickle.dumps((args, generation, action, point)))
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "tests.environments.prepared_dataset_workload",
                            str(config),
                        ],
                        env={
                            **os.environ,
                            "J41_READY_FD": str(ready_w),
                            "J41_RELEASE_FD": str(release_r),
                            "J41_CONSUMER_FD": str(consumer_w),
                        },
                        pass_fds=(ready_w, release_r, consumer_w),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                fd, marker = (
                    (ready_r, b"R") if action == "produce" else (consumer_r, b"T")
                )
                assert select.select([fd], [], [], 60)[0]
                assert os.read(fd, 1) == marker
            key, record = one(paths)
            assert record["phase"] == "PREPARING"
            assert not prepared.retire(paths, key)
            # Both children are admitted; bulk preparation and its waiter own no tree lock.
            with growth_receipt_ownership(paths):
                pass
            os.write(release_w, b"G")
            for process in children:
                assert process.wait(timeout=90) == 0, (
                    paths.work_dir / "race.log"
                ).read_text()
            assert one(paths)[1]["phase"] == "COMPLETE"
            assert prepared.retire(paths, key)
        finally:
            for process in children:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            for fd in (ready_r, ready_w, release_r, release_w, consumer_r, consumer_w):
                os.close(fd)
