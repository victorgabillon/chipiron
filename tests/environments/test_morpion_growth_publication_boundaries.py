"""Private preparation, deferred prerequisites and late Reevaluation receipts."""

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.config import (
    load_bootstrap_config,
    save_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.generation_publication import (
    DurableGenerationPublication,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    capsule_path,
    growth_receipt_ownership,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    TRANSACTION_KEY,
    cursor_digest,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_atomic_reevaluation_helpers import (
    accepted,
    publish_proposal,
    setup_atomic,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
)
from tests.environments.morpion_patch_receipt_helpers import BASELINE, PATCHED
from tests.environments.test_morpion_atomic_reevaluation import checkpoint

if TYPE_CHECKING:
    from pathlib import Path


def test_patch_arriving_during_growth_remains_until_checkpoint_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real create-only proposal cannot advance the cursor of an unpatched tree."""
    args, paths, _ = setup_atomic(tmp_path)
    runner = ClosedPublicationRunner()
    # This fixture normally has no growth and unreachable save thresholds.
    # Add its supported third node and trigger a save before the late patch applies.
    runner.grow_delta = 1
    monkeypatch.setattr(stages, "should_save_progress", lambda **_: True)
    original = runner.grow
    proposals = []

    def arrived(*a: Any, **kw: Any) -> Any:
        result = original(*a, **kw)
        with growth_receipt_ownership(paths):
            proposals.append(publish_proposal(paths))
        return result

    monkeypatch.setattr(runner, "grow", arrived)
    first = run_closed_publication(args, runner)
    assert first.generation == 2 and runner.apply_calls == 0
    assert checkpoint(paths, first) == {**BASELINE, "new": 0.0}
    assert accepted(first) is None and receipt_ledger(first) == {}
    assert paths.pipeline_reevaluation_patch_path.exists()
    next_runner = ClosedPublicationRunner()
    second = run_closed_publication(args, next_runner)
    assert second.generation == 3 and next_runner.apply_calls == 1
    [proposal] = proposals
    receipt = receipt_ledger(second)[proposal.patch_id]
    assert checkpoint(paths, second) == {**PATCHED, "new": 0.0}
    assert accepted(second) == proposal.metadata[TRANSACTION_KEY]["next_cursor"]
    assert receipt.accepted_cursor_sha256 == cursor_digest(accepted(second))
    assert receipt.checkpoint_path == second.latest_runtime_checkpoint_path
    assert not paths.pipeline_reevaluation_patch_path.exists()
    cold = ClosedPublicationRunner()
    assert run_closed_publication(args, cold, max_cycles=0) == second
    assert cold.apply_calls == 0


def test_changed_configuration_discards_private_result_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed admitted input never commits its checkpoint or applied cursor."""
    args, paths, source = setup_atomic(tmp_path)
    proposal = publish_proposal(paths)
    config = load_bootstrap_config(paths.bootstrap_config_path)
    original = GrowthPatchReceiptSession.prepare_growth

    def changed(self: Any, update: Any) -> None:
        original(self, update)
        with growth_receipt_ownership(paths):
            save_bootstrap_config(
                replace(
                    config,
                    runtime=replace(
                        config.runtime,
                        save_after_seconds=config.runtime.save_after_seconds + 1,
                    ),
                ),
                paths.bootstrap_config_path,
            )

    with monkeypatch.context() as patch:
        patch.setattr(GrowthPatchReceiptSession, "prepare_growth", changed)
        assert run_closed_publication(args, ClosedPublicationRunner()) == source
    assert load_bootstrap_run_state(paths.run_state_path) == source
    assert accepted(source) is None and paths.pipeline_reevaluation_patch_path.exists()
    with growth_receipt_ownership(paths):
        save_bootstrap_config(config, paths.bootstrap_config_path)
    runner = ClosedPublicationRunner()
    retried = run_closed_publication(args, runner)
    assert retried.generation == 2 and runner.apply_calls == 1
    assert checkpoint(paths, retried) == PATCHED
    assert set(receipt_ledger(retried)) == {proposal.patch_id}


def test_prepared_capsule_change_is_rejected_before_authority_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The volatile fingerprint never accepts a modified previously hashed result."""
    args, paths, source = setup_atomic(tmp_path)
    # Applying a real pending patch requires a checkpoint even with no new nodes.
    proposal = publish_proposal(paths)
    original = DurableGenerationPublication.prepare_capsule

    def changed(self: Any, *a: Any, **kw: Any) -> Any:
        result = original(self, *a, **kw)
        final = capsule_path(self.paths.work_dir, self.identifier, self.version)
        (final / "unexpected-after-seal").write_text("changed")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(DurableGenerationPublication, "prepare_capsule", changed)
        with pytest.raises(ValueError, match="Prepared capsule changed"):
            run_closed_publication(args, ClosedPublicationRunner())
    assert load_bootstrap_run_state(paths.run_state_path) == source
    assert paths.pipeline_reevaluation_patch_path.exists() and accepted(source) is None
    retried = run_closed_publication(args, ClosedPublicationRunner())
    assert retried.generation == 2 and checkpoint(paths, retried) == PATCHED
    assert set(receipt_ledger(retried)) == {proposal.patch_id}


def test_new_hardlink_revalidates_sealed_bytes_without_false_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inode ctime is a fast fence, not proof that immutable payload bytes changed."""
    args, paths, _ = setup_atomic(tmp_path)
    proposal = publish_proposal(paths)
    original = DurableGenerationPublication.prepare_capsule

    def linked(self: Any, *a: Any, **kw: Any) -> Any:
        state, clocks = original(self, *a, **kw)
        os.link(
            paths.work_dir / state.latest_runtime_checkpoint_path,
            tmp_path / "reader-link",
        )
        return state, clocks

    monkeypatch.setattr(DurableGenerationPublication, "prepare_capsule", linked)
    state = run_closed_publication(args, ClosedPublicationRunner())
    assert state.generation == 2
    assert checkpoint(paths, state) == PATCHED
    assert set(receipt_ledger(state)) == {proposal.patch_id}
    assert (paths.work_dir / state.latest_runtime_checkpoint_path).read_bytes() == (
        tmp_path / "reader-link"
    ).read_bytes()
