"""Helpers for choosing SearchDynamics adapters from runtime environment dynamics."""

from typing import Any

from anemone.dynamics import SearchDynamics, normalize_search_dynamics
from valanga import Dynamics

from chipiron.environments.chess.search_dynamics import ChessSearchDynamics
from chipiron.environments.types import GameKind


def make_search_dynamics(
    *,
    game_kind: GameKind,
    dynamics: Dynamics[Any],
    copy_stack_until_depth: int = 2,
    deep_copy_legal_moves: bool = True,
) -> SearchDynamics[Any, Any]:
    """Build search dynamics for Anemone from game runtime dynamics."""
    if game_kind is GameKind.CHESS:
        return ChessSearchDynamics(
            copy_stack_until_depth=copy_stack_until_depth,
            deep_copy_legal_moves=deep_copy_legal_moves,
        )

    return normalize_search_dynamics(dynamics)
