"""Existing restore interface, continuation and decoded-state retention contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from anemone.checkpoints.rng_state import rollout_random_generator

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.config import MorpionBootstrapRolloutConfig
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from docs.research.corrected_anemone_refresh.test_integration import refresh
from docs.research.model_activation.measure import fingerprint, link_iteration_digest
from docs.research.model_activation.probe import graph, nodes, state
from docs.research.rolling_snapshot.measure import runner
from tests.environments.test_morpion_checkpoint_parity import _selection_signature
from tests.environments.test_morpion_value_certainty import neural_evaluator

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("format_name", ["json_zst", "sharded"])
@pytest.mark.parametrize("rollout", [False, True])
def test_committed_evaluator_supplied_at_load_preserves_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    rollout: bool,
) -> None:
    """Supply identical known weights through Anemone's loader, with no set call."""
    args = replace(runner()._args, runtime_checkpoint_format=format_name)
    if rollout:
        args = replace(
            args,
            search_args=runner_module.default_search_args(
                rollout=MorpionBootstrapRolloutConfig(enabled=True, max_extra_steps=1)
            ),
        )
    current = runner_module.AnemoneMorpionSearchRunner(args)
    config = MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    current.load_or_create(None, None, config)
    current.grow(3)
    evaluator = neural_evaluator(42)
    rng = current._random_generator.getstate()
    rollout_rng = rollout_random_generator(current._runtime)
    rollout_before = None if rollout_rng is None else rollout_rng.getstate()
    refresh(current, evaluator)
    assert current._random_generator.getstate() == rng
    assert (None if rollout_rng is None else rollout_rng.getstate()) == rollout_before
    path = tmp_path / (
        "checkpoint" if format_name == "sharded" else "checkpoint.json.zst"
    )
    current.save_checkpoint(path)
    restored = runner_module.AnemoneMorpionSearchRunner(args)
    # The runner currently asks for its default evaluator at load. Inject the exact
    # known model there to exercise Anemone's EXISTING public restore interface.
    monkeypatch.setattr(restored, "_build_master_evaluator", lambda _: evaluator)
    restored.load_or_create(path, None, config)
    assert fingerprint(restored) == fingerprint(current)
    assert restored._runtime.evaluator_version == current._runtime.evaluator_version
    restored_rollout = rollout_random_generator(restored._runtime)
    assert (
        None if restored_rollout is None else restored_rollout.getstate()
    ) == rollout_before
    (tmp_path / "checkpoint-order.json").write_text(
        json.dumps(
            {
                "scientific_fingerprints_equal": True,
                "ordered_links_equal": link_iteration_digest(current)
                == link_iteration_digest(restored),
                "representative_parent_discrepancy_fixed": False,
                "production_resume_binding_implemented": False,
            },
            indent=2,
        )
        + "\n"
    )
    for _ in range(3):
        current.grow(1)
        restored.grow(1)
        assert state(restored) == state(current)
        assert fingerprint(restored) == fingerprint(current)
        assert _selection_signature(restored) == _selection_signature(current)
        assert (
            restored._random_generator.getstate()
            == current._random_generator.getstate()
        )
        if rollout:
            assert restored_rollout.getstate() == rollout_rng.getstate()


def test_existing_runner_bundle_resume_still_increments_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterize the remaining caller defect, rather than claiming it fixed."""
    current = graph()
    evaluator = neural_evaluator(42)
    refresh(current, evaluator)
    path = tmp_path / "checkpoint"
    current.save_checkpoint(path)
    monkeypatch.setattr(
        runner_module, "load_morpion_evaluator_from_model_bundle", lambda _: evaluator
    )
    restored = runner()
    restored.load_or_create(path, tmp_path / "known-same-model")
    assert restored._runtime.evaluator_version == current._runtime.evaluator_version + 1
    assert all(
        n.tree_evaluation.direct_evaluation_version
        == current._runtime.evaluator_version
        for n in nodes(restored)
    )


@pytest.mark.parametrize("batch_size", [1, 64])
def test_restore_cache_still_retains_every_decoded_state(
    tmp_path: Path, batch_size: int
) -> None:
    """Observe the existing resolver, without clearing or replacing its cache."""
    current = graph()
    path = tmp_path / "checkpoint"
    current.save_checkpoint(path)
    restored = runner()
    restored.load_or_create(path, None)
    resolver = next(nodes(restored)).tree_node.state_handle.resolver
    before = len(resolver._resolved_states)
    report = refresh(restored, neural_evaluator(42), batch_size)
    after = len(resolver._resolved_states)
    assert before < after == restored.current_tree_size()
    (tmp_path / "cache-retention.json").write_text(
        json.dumps(
            dict(
                batch_size=batch_size,
                before=before,
                after=after,
                requested=report.requested_count,
            ),
            indent=2,
        )
        + "\n"
    )
