"""
Module for building game state evaluators (oracle + chipiron) with optional GUI publishing.
"""

from typing import Any, Literal, TypeVar, assert_never, overload

from chipiron.environments.chess.types import ChessState
from chipiron.environments.types import GameKind
from chipiron.players.boardevaluators.wirings.chess_eval_wiring import ChessEvalWiring
from chipiron.players.boardevaluators.wirings.null_eval_wiring import NullEvalWiring
from chipiron.players.boardevaluators.wirings.protocols import EvaluatorWiring

from .board_evaluator import (
    GameStateEvaluator,
    IGameStateEvaluator,
    ObservableGameStateEvaluator,
)

StateT = TypeVar("StateT")


@overload
def _select_eval_wiring(
    game_kind: Literal[GameKind.CHESS], *, can_oracle: bool
) -> EvaluatorWiring[ChessState]: ...


@overload
def _select_eval_wiring(
    game_kind: Literal[GameKind.CHECKERS], *, can_oracle: bool
) -> EvaluatorWiring[object]: ...


@overload
def _select_eval_wiring(
    game_kind: GameKind, *, can_oracle: bool
) -> EvaluatorWiring[Any]: ...


def _select_eval_wiring(
    game_kind: GameKind, *, can_oracle: bool
) -> EvaluatorWiring[Any]:
    """Select the evaluator wiring for the requested game kind."""
    match game_kind:
        case GameKind.CHESS:
            return ChessEvalWiring(can_oracle=can_oracle)
        case GameKind.CHECKERS:
            return NullEvalWiring()
        case _:
            assert_never(game_kind)


def create_game_board_evaluator(
    *,
    wiring: EvaluatorWiring[StateT],
    gui: bool,
) -> IGameStateEvaluator[StateT]:
    """Create game board evaluator."""
    base: IGameStateEvaluator[StateT] = GameStateEvaluator(
        chi=wiring.build_chi(),
        oracle=wiring.build_oracle(),
    )

    if gui:
        return ObservableGameStateEvaluator(base)

    return base


@overload
def create_game_board_evaluator_for_game_kind(
    *, game_kind: Literal[GameKind.CHESS], gui: bool, can_oracle: bool
) -> IGameStateEvaluator[ChessState]: ...


@overload
def create_game_board_evaluator_for_game_kind(
    *, game_kind: Literal[GameKind.CHECKERS], gui: bool, can_oracle: bool
) -> IGameStateEvaluator[object]: ...


@overload
def create_game_board_evaluator_for_game_kind(
    *, game_kind: GameKind, gui: bool, can_oracle: bool
) -> IGameStateEvaluator[Any]: ...


def create_game_board_evaluator_for_game_kind(
    *,
    game_kind: GameKind,
    gui: bool,
    can_oracle: bool,
) -> IGameStateEvaluator[Any]:
    """Create game board evaluator for game kind."""
    wiring = _select_eval_wiring(game_kind, can_oracle=can_oracle)
    return create_game_board_evaluator(
        wiring=wiring,
        gui=gui,
    )
