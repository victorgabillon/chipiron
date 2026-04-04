"""Runtime-configurable addons for search dynamics wrappers."""

import importlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from anemone.dynamics import SearchDynamics
from valanga import TurnState

type AnyTurnState = TurnState


class SearchDynamicsAddonType(StrEnum):
    """Supported optional search dynamics wrappers."""

    CHESS_COPY_STACK = "CHESS_COPY_STACK"


@dataclass(frozen=True)
class ChessCopyStackAddonArgs:
    """Configuration for depth-aware chess board stack copying."""

    type: SearchDynamicsAddonType = SearchDynamicsAddonType.CHESS_COPY_STACK
    copy_stack_until_depth: int = 2
    deep_copy_legal_moves: bool = True


SearchDynamicsAddonArgs = ChessCopyStackAddonArgs


def apply_search_dynamics_addon[TurnStateT: AnyTurnState](
    *,
    addon: SearchDynamicsAddonArgs,
    state_type: type[TurnStateT],
    search_dynamics: SearchDynamics[TurnStateT, Any],
) -> SearchDynamics[TurnStateT, Any]:
    """Apply an optional game-specific search dynamics wrapper."""
    if addon.type is SearchDynamicsAddonType.CHESS_COPY_STACK:
        chess_search_dynamics_module = importlib.import_module(
            "chipiron.environments.chess.search_dynamics"
        )
        chess_types_module = importlib.import_module(
            "chipiron.environments.chess.types"
        )
        chess_copy_stack_search_dynamics = (
            chess_search_dynamics_module.ChessCopyStackSearchDynamics
        )
        chess_state = chess_types_module.ChessState

        try:
            is_chess = issubclass(state_type, chess_state)
        except TypeError:
            is_chess = False

        if is_chess:
            chess_base = cast("SearchDynamics[Any, Any]", search_dynamics)
            wrapped = chess_copy_stack_search_dynamics(
                base=chess_base,
                copy_stack_until_depth=addon.copy_stack_until_depth,
                deep_copy_legal_moves=addon.deep_copy_legal_moves,
            )
            return cast("SearchDynamics[TurnStateT, Any]", wrapped)

    return search_dynamics
