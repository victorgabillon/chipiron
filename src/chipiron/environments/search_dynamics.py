"""Helpers for choosing SearchDynamics adapters from runtime environment dynamics."""

from typing import Any

from anemone.dynamics import SearchDynamics, normalize_search_dynamics
from valanga import Dynamics

from chipiron.environments.types import GameKind


def make_search_dynamics(
    *,
    game_kind: GameKind,
    dynamics: Dynamics[Any],
) -> SearchDynamics[Any, Any]:
    """Build search dynamics for Anemone from game runtime dynamics."""
    _ = game_kind
    return normalize_search_dynamics(dynamics)
