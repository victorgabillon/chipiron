"""Tests for anemone hooks wiring in move selector factory."""

import random
from typing import Any, cast

from anemone.hooks.search_hooks import SearchHooks

from chipiron.environments.chess.types import ChessState
from chipiron.players.move_selector import factory
from chipiron.players.move_selector.anemone_hooks import ChessFeatureExtractor


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
        args=cast(Any, object()),
        state_type=ChessState,
        master_state_evaluator=cast(Any, object()),
        state_representation_factory=None,
        random_generator=random.Random(0),
    )

    assert selector is sentinel_selector
    hooks = captured.get("hooks")
    assert isinstance(hooks, SearchHooks)
    assert isinstance(hooks.feature_extractor, ChessFeatureExtractor)
