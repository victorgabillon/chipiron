"""Negative boundaries use real model-only workspaces and existing admission gates."""

from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import model_activation as activation
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
)
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.test_morpion_progressive_activation import setup

if TYPE_CHECKING:
    from pathlib import Path


def model_only_factory(root: Path, **kwargs: Any) -> Any:
    """Select the real fresh mode for existing refusal scenarios."""
    return setup_activation(root, model_only_activation=True, **kwargs)


@pytest.mark.parametrize(
    "damage",
    [
        "weights",
        "missing",
        "pending",
        "fallback",
        "checkpoint",
        "science",
        "no_training",
    ],
)
def test_real_activation_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Reuse exact historical corruption gates without changing their assertions."""
    from tests.environments import test_morpion_model_activation_edges as edges

    monkeypatch.setattr(edges, "setup_activation", model_only_factory)
    edges.test_corruption_and_unsupported_proposals_block(tmp_path, monkeypatch, damage)


@pytest.mark.parametrize("damage", ["missing", "changed", "uncommitted"])
def test_real_mixed_origin_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """The production mixed path cannot replace genuine sidecar authentication."""
    from tests.environments import test_morpion_prediction_origin_writes as origins

    monkeypatch.setattr(origins, "setup_activation", model_only_factory)
    origins.test_committed_reader_refuses_broken_or_uncommitted_evidence(
        tmp_path, monkeypatch, damage
    )


def test_private_switch_failure_discards_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource failure after the switch cannot publish partially changed RAM."""
    _, paths, _ = setup(tmp_path, monkeypatch)
    request = activation.prepare_model_activation_request(paths, training_generation=1)
    before = paths.run_state_path.read_bytes()
    original = AnemoneMorpionSearchRunner.switch_model_evaluator
    instances = []

    def fail(self: AnemoneMorpionSearchRunner, path: Path) -> None:
        original(self, path)
        instances.append(self)
        raise MemoryError

    monkeypatch.setattr(AnemoneMorpionSearchRunner, "switch_model_evaluator", fail)
    with pytest.raises(MemoryError):
        activation.activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == before
    assert instances and all(r._runtime is None for r in instances)


def test_inconsistent_certificate_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an owned private runtime may not promote contradictory proof values."""
    from anemone.node_evaluation.common.canonical_value import make_forced_value

    _, paths, _ = setup(tmp_path, monkeypatch)
    request = activation.prepare_model_activation_request(paths, training_generation=1)
    before = paths.run_state_path.read_bytes()
    original = AnemoneMorpionSearchRunner.load_or_create

    def damaged(self: AnemoneMorpionSearchRunner, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self._require_runtime().tree.root_node.tree_evaluation.direct_value = (
            make_forced_value(score=999)
        )

    monkeypatch.setattr(AnemoneMorpionSearchRunner, "load_or_create", damaged)
    with pytest.raises(ValueError):
        activation.activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == before


def test_competing_model_only_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold the real switch under Growth ownership while a competing caller tries."""
    from threading import Event, Thread

    _, paths, _ = setup(tmp_path, monkeypatch)
    request = activation.prepare_model_activation_request(paths, training_generation=1)
    arrived, release = Event(), Event()
    original = AnemoneMorpionSearchRunner.switch_model_evaluator
    results: dict[str, Any] = {}

    def pause(self: AnemoneMorpionSearchRunner, path: Path) -> None:
        arrived.set()
        assert release.wait(30)
        original(self, path)

    def run() -> None:
        try:
            results["state"] = activation.activate_model(paths, request)
        except Exception as error:
            results["error"] = error

    monkeypatch.setattr(AnemoneMorpionSearchRunner, "switch_model_evaluator", pause)
    thread = Thread(target=run)
    thread.start()
    try:
        assert arrived.wait(30), results
        with pytest.raises(BlockingIOError):
            activation.activate_model(paths, request)
    finally:
        release.set()
        thread.join(30)
    assert not thread.is_alive() and "error" not in results, results
    assert activation.activate_model(paths, request) == results["state"]
