"""Atomic patch/cursor acceptance, explicit rejection, exact retry and scientific parity."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
    publish_reevaluation_patch_if_absent,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    TRANSACTION_KEY,
    cursor_digest,
)
from tests.environments.morpion_atomic_reevaluation_helpers import (
    accepted,
    make_proposal,
    publish_proposal,
    setup_atomic,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
)
from tests.environments.morpion_patch_receipt_helpers import BASELINE, PATCHED

if TYPE_CHECKING:
    from pathlib import Path


def checkpoint(paths: Any, state: Any) -> Any:
    """Read the tiny scientific fixture's complete scalar checkpoint."""
    return json.loads(
        (paths.work_dir / state.latest_runtime_checkpoint_path).read_text()
    )


def test_committed_patch_has_cursor_without_singleton(tmp_path: Path) -> None:
    """The old missing-cursor stall is impossible under the explicit protocol."""
    args, paths, _ = setup_atomic(tmp_path)
    patch = publish_proposal(paths)
    runner = ClosedPublicationRunner()
    state = run_closed_publication(args, runner)
    assert runner.apply_calls == 1 and checkpoint(paths, state) == PATCHED
    assert accepted(state) == patch.metadata[TRANSACTION_KEY]["next_cursor"]
    assert receipt_ledger(state)[
        patch.patch_id
    ].accepted_cursor_sha256 == cursor_digest(accepted(state))
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()
    for _ in range(3):
        runner = ClosedPublicationRunner()
        assert run_closed_publication(args, runner, max_cycles=0) == state
        assert runner.apply_calls == 0


def test_old_exact_retry_never_rewinds_newer_cursor(tmp_path: Path) -> None:
    """Lost acknowledgement of an older exact proposal cannot regress progress."""
    args, paths, _ = setup_atomic(tmp_path)
    publish_proposal(paths)
    original = paths.pipeline_reevaluation_patch_path.read_bytes()
    run_closed_publication(args, ClosedPublicationRunner())
    publish_proposal(paths, identifier="proposal-2")
    state = run_closed_publication(args, ClosedPublicationRunner())
    before = paths.run_state_path.read_bytes()
    assert accepted(state)["completed_full_pass_count"] == 2
    paths.pipeline_reevaluation_patch_path.write_bytes(original)
    runner = ClosedPublicationRunner()
    recovered = run_closed_publication(args, runner, max_cycles=0)
    assert runner.apply_calls == 0 and recovered == state
    assert paths.run_state_path.read_bytes() == before
    assert not paths.pipeline_reevaluation_patch_path.exists()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("wrap", [False, True])
def test_empty_and_wrap_transitions_commit_once(
    tmp_path: Path, empty: bool, wrap: bool
) -> None:
    """Cursor-only work uses exactly the same checkpoint/receipt boundary."""
    args, paths, _ = setup_atomic(tmp_path)
    publish_proposal(paths, empty=empty, wrap=wrap)
    state = run_closed_publication(args, ClosedPublicationRunner())
    assert state.generation == 2
    assert checkpoint(paths, state) == (BASELINE if empty else PATCHED)
    assert accepted(state)["completed_full_pass_count"] == int(wrap)
    assert accepted(state)["next_node_cursor"] == (None if wrap else "root")
    assert receipt_ledger(state)["proposal-1"].rows_applied == (0 if empty else 2)


@pytest.mark.parametrize(
    "field",
    [
        "protocol",
        "schema_version",
        "workspace_id",
        "epoch_id",
        "producer_id",
        "source",
        "model",
        "expected_cursor_sha256",
        "next_cursor",
        "missing_counter",
        "bad_counter",
        "missing_protocol",
    ],
)
def test_malformed_foreign_or_stale_proposals_fail_before_apply(
    tmp_path: Path, field: str
) -> None:
    """Invalid continuation evidence never changes committed values or receipts."""
    args, paths, _ = setup_atomic(tmp_path)
    patch = make_proposal(paths)
    metadata = copy.deepcopy(patch.metadata)
    envelope = metadata[TRANSACTION_KEY]
    if field == "missing_counter":
        del envelope["next_cursor"]["completed_full_pass_count"]
    elif field == "bad_counter":
        envelope["next_cursor"]["completed_full_pass_count"] += 1
    elif field == "missing_protocol":
        del metadata[TRANSACTION_KEY]
    else:
        envelope[field] = "invalid"
    patch = replace(patch, metadata=metadata)
    assert publish_reevaluation_patch_if_absent(
        patch, paths.pipeline_reevaluation_patch_path
    )
    before = paths.run_state_path.read_bytes()
    runner = ClosedPublicationRunner()
    with pytest.raises((ValueError, TypeError)):
        run_closed_publication(args, runner)
    assert runner.apply_calls == 0 and paths.run_state_path.read_bytes() == before
    assert paths.pipeline_reevaluation_patch_path.exists()


def test_competing_proposals_and_conflicting_uuid(tmp_path: Path) -> None:
    """Create-only wins once; a new stale UUID and conflicting duplicate both fail."""
    args, paths, _ = setup_atomic(tmp_path)
    first = make_proposal(paths)
    loser = make_proposal(paths, "competitor")
    assert publish_reevaluation_patch_if_absent(
        first, paths.pipeline_reevaluation_patch_path
    )
    assert not publish_reevaluation_patch_if_absent(
        loser, paths.pipeline_reevaluation_patch_path
    )
    run_closed_publication(args, ClosedPublicationRunner())
    for proposal, message in [
        (loser, "Stale expected"),
        (
            replace(first, metadata={**first.metadata, "extra": True}),
            "Conflicting reused",
        ),
    ]:
        assert publish_reevaluation_patch_if_absent(
            proposal, paths.pipeline_reevaluation_patch_path
        )
        before = paths.run_state_path.read_bytes()
        runner = ClosedPublicationRunner()
        with pytest.raises(ValueError, match=message):
            run_closed_publication(args, runner)
        assert runner.apply_calls == 0 and paths.run_state_path.read_bytes() == before
        paths.pipeline_reevaluation_patch_path.unlink()  # Rejected synthetic input only.


@pytest.mark.parametrize(
    "damage", ["cursor", "counter", "receipt_digest", "missing_protocol"]
)
def test_corrupt_committed_cursor_fails_closed(tmp_path: Path, damage: str) -> None:
    """Recovery never invents missing state from a singleton or a receipt count."""
    args, paths, _ = setup_atomic(tmp_path)
    publish_proposal(paths)
    run_closed_publication(args, ClosedPublicationRunner())
    data = json.loads(paths.run_state_path.read_text())
    metadata = data["metadata"]
    if damage == "cursor":
        metadata[TRANSACTION_KEY]["accepted_cursor"] = None
    elif damage == "counter":
        del metadata[TRANSACTION_KEY]["accepted_cursor"]["completed_full_pass_count"]
    elif damage == "receipt_digest":
        metadata["checkpoint_bound_reevaluation_v1"]["receipts"][0][
            "accepted_cursor_sha256"
        ] = "0" * 64
    else:
        del metadata[TRANSACTION_KEY]
    paths.run_state_path.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        run_closed_publication(args, ClosedPublicationRunner(), max_cycles=0)


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_resumes_committed_position_without_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """The real producer reads only accepted state, including a complete wrap."""
    from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        load_closed_scientific_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_reevaluation_patch,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView,
    )
    from tests.environments.test_morpion_scientific_training_reader import StateScore

    args, paths, _ = setup_atomic(tmp_path)

    def evaluator(**_kwargs: Any) -> Any:
        result = worker.MorpionActiveModelNodeReevaluationEvaluator(
            tmp_path / "unused-model"
        )
        result._evaluator = StateScore()
        return result

    monkeypatch.setattr(worker, "build_active_model_reevaluation_evaluator", evaluator)
    monkeypatch.setattr(worker, "is_sharded_generation_manifest", lambda _: True)
    monkeypatch.setattr(
        worker,
        "ShardedTrainingGenerationView",
        partial(ShardedTrainingGenerationView, allow_closed_snapshot=True),
    )
    monkeypatch.setattr(
        worker,
        "load_reevaluation_training_tree_snapshot",
        lambda p: load_closed_scientific_snapshot(p).snapshot,
    )
    for number in range(3):
        before = paths.run_state_path.read_bytes()
        result = worker.run_morpion_reevaluation_worker_once(
            args,
            max_nodes_per_patch=1,
            patch_id=f"worker-{number}",
            now_unix_s=1e9,
            use_streaming_sharded_reader=streaming,
            allow_scientific_v2=True,
        )
        assert result.patch_written
        assert paths.run_state_path.read_bytes() == before
        patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
        assert patch.start_cursor == ("root" if number == 1 else "leaf")
        assert not paths.pipeline_reevaluation_cursor_path.exists()
        state = run_closed_publication(args, ClosedPublicationRunner())
        assert accepted(state) == patch.metadata[TRANSACTION_KEY]["next_cursor"]
        assert accepted(state)["completed_full_pass_count"] == (number + 1) // 2


def test_full_and_bounded_entire_proposal_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold identities fixed and compare every scientific row and proposal byte."""
    from tests.environments.morpion_atomic_reevaluation_helpers import atomic_worker

    args, paths, _ = setup_atomic(tmp_path)
    expected = atomic_worker(tmp_path, monkeypatch, streaming=False)
    raw = paths.pipeline_reevaluation_patch_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.unlink()  # Unaccepted generated proposal only.
    actual = atomic_worker(tmp_path, monkeypatch, streaming=True)
    assert actual == expected
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == raw
    state = run_closed_publication(args, ClosedPublicationRunner())
    assert accepted(state)["last_patch_id"] == "worker-proposal"


def test_captured_epoch_is_not_retagged_at_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthetic mid-computation retag cannot bless an old result in the new epoch."""
    from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
    from tests.environments.morpion_atomic_reevaluation_helpers import atomic_worker

    args, paths, initial = setup_atomic(tmp_path)
    old_proposal = make_proposal(paths)
    original = worker.MorpionActiveModelNodeReevaluationEvaluator.evaluate_patch_rows

    def evaluate(evaluator: Any, *positional: Any, **kwargs: Any) -> Any:
        data = json.loads(paths.run_state_path.read_text())
        data["metadata"][TRANSACTION_KEY]["epoch_id"] = "1" * 32
        paths.run_state_path.write_text(json.dumps(data))
        return original(evaluator, *positional, **kwargs)

    monkeypatch.setattr(
        worker.MorpionActiveModelNodeReevaluationEvaluator,
        "evaluate_patch_rows",
        evaluate,
    )
    with pytest.raises(ValueError, match="epoch/producer mismatch"):
        atomic_worker(tmp_path, monkeypatch, streaming=False)
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()
    # A stale proposal from an older producer must still be rejected by Growth.
    assert publish_reevaluation_patch_if_absent(
        old_proposal, paths.pipeline_reevaluation_patch_path
    )
    raw = json.loads(paths.pipeline_reevaluation_patch_path.read_text())
    assert (
        raw["metadata"][TRANSACTION_KEY]["epoch_id"]
        == initial.metadata[TRANSACTION_KEY]["epoch_id"]
    )
    runner = ClosedPublicationRunner()
    with pytest.raises(ValueError, match="epoch/producer mismatch"):
        run_closed_publication(args, runner)
    assert runner.apply_calls == 0


@pytest.mark.parametrize("change", ["model_bytes", "model_identity", "source_bytes"])
def test_source_or_model_change_rejected(tmp_path: Path, change: str) -> None:
    """Reusing a path/generation cannot authorize different scientific dependencies."""
    args, paths, initial = setup_atomic(tmp_path)
    publish_proposal(paths)
    if change == "model_bytes":
        (paths.work_dir / "models/generation_000001/default/param.pt").write_bytes(
            b"different"
        )
    elif change == "model_identity":
        data = json.loads(paths.pipeline_active_model_path.read_text())
        data["evaluator_name"] = "changed"
        paths.pipeline_active_model_path.write_text(json.dumps(data))
    else:
        (paths.work_dir / initial.latest_tree_snapshot_path).write_text("{}")
    before = paths.run_state_path.read_bytes()
    runner = ClosedPublicationRunner()
    with pytest.raises(ValueError):
        run_closed_publication(args, runner)
    assert runner.apply_calls == 0 and paths.run_state_path.read_bytes() == before


def test_concurrent_growth_ownership(tmp_path: Path) -> None:
    """Existing stable exclusive ownership remains the sole consumer lock."""
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        growth_receipt_ownership,
    )

    args, paths, _ = setup_atomic(tmp_path)
    publish_proposal(paths)
    with growth_receipt_ownership(paths), pytest.raises(BlockingIOError):
        run_closed_publication(args, ClosedPublicationRunner())
    assert run_closed_publication(args, ClosedPublicationRunner()).generation == 2
