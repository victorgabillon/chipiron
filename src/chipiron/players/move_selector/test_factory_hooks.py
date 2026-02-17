"""Tests for anemone hooks wiring in move selector factory."""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import valanga
from anemone.dynamics import SearchDynamics
from anemone.hooks.search_hooks import SearchHooks

from chipiron.environments.chess.types import ChessState
from chipiron.players.move_selector import factory
from chipiron.players.move_selector.anemone_hooks import ChessFeatureExtractor
from chipiron.players.move_selector.priority_checks.pv_attacked_open_all import (
    PvAttackedOpenAllPriorityCheck,
)

if TYPE_CHECKING:
    from anemone.node_selector.priority_check.priority_check import PriorityCheck


class NotUsedInWiringTestError(AssertionError):
    """Exception raised when a test stub method is unexpectedly called."""

    def __init__(self) -> None:
        """Initialize with a descriptive message."""
        super().__init__("Not used in this wiring test")


class DummyOpeningInstructor:
    """Minimal opening instructor stub for factory wiring tests."""

    def all_branches_to_open(self, node: Any) -> list[Any]:
        """Return empty list for all nodes."""
        _ = node
        return []


@dataclass(frozen=True)
class DummySearchDynamics(SearchDynamics[ChessState, valanga.BranchKey]):
    """Bare-minimum SearchDynamics for hook wiring tests."""

    __anemone_search_dynamics__ = True

    def legal_actions(
        self, state: ChessState
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]:
        """Return legal actions for a state (unused in this test stub)."""
        _ = state
        raise NotUsedInWiringTestError

    def step(
        self, state: ChessState, action: valanga.BranchKey, *, depth: int
    ) -> valanga.Transition[ChessState]:
        """Step dynamics for one action (unused in this test stub)."""
        _ = (state, action, depth)
        raise NotUsedInWiringTestError

    def action_name(self, state: ChessState, action: valanga.BranchKey) -> str:
        """Return action string name (unused in this test stub)."""
        _ = (state, action)
        raise NotUsedInWiringTestError

    def action_from_name(self, state: ChessState, name: str) -> valanga.BranchKey:
        """Parse action string into key (unused in this test stub)."""
        _ = (state, name)
        raise NotUsedInWiringTestError


def test_create_tree_and_value_move_selector_passes_hooks(monkeypatch: Any) -> None:
    """Ensure SearchHooks are constructed and forwarded to anemone factory."""
    captured: dict[str, Any] = {}
    sentinel_selector = object()

    def fake_create_tree_and_value_branch_selector(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel_selector

    monkeypatch.setattr(
        factory,
        "create_tree_and_value_branch_selector",
        fake_create_tree_and_value_branch_selector,
    )

    selector = factory.create_tree_and_value_move_selector(
        args=cast("Any", object()),
        state_type=ChessState,
        master_state_evaluator=cast("Any", object()),
        state_representation_factory=None,
        random_generator=random.Random(0),
        search_dynamics_override=DummySearchDynamics(),
    )

    assert selector is sentinel_selector
    hooks = captured.get("hooks")
    assert isinstance(hooks, SearchHooks)
    assert isinstance(hooks.feature_extractor, ChessFeatureExtractor)
    assert "pv_attacked_open_all" in hooks.priority_check_registry

    priority_check_factory = hooks.priority_check_registry["pv_attacked_open_all"]

    priority_check: PriorityCheck = priority_check_factory(
        {"probability": 0.9, "feature_key": "tactical_threat"},  # params
        random.Random(0),  # random_generator
        hooks,  # hooks
        cast("Any", DummyOpeningInstructor()),  # opening_instructor
    )

    assert isinstance(priority_check, PvAttackedOpenAllPriorityCheck)
    assert priority_check.probability == 0.9
    assert priority_check.feature_key == "tactical_threat"
