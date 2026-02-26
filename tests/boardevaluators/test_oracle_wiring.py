"""Module for test oracle wiring."""

import queue

import chess
import pytest
from atomheart.games.chess.board import IBoard, create_board
from atomheart.games.chess.board.utils import FenPlusHistory
from valanga import StateEvaluation

from chipiron.displays.gui_protocol import GuiUpdate, UpdEvaluation, make_scope
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.environments.chess.types import ChessState
from chipiron.environments.types import GameKind
from chipiron.players.boardevaluators.board_evaluator import (
    ObservableGameStateEvaluator,
)
from chipiron.environments.chess.players.evaluators.boardevaluators.factory import (
    create_game_board_evaluator_for_game_kind,
)


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
def test_chess_evaluator_accepts_chess_state(use_rust_boards: bool) -> None:
    """Test chess evaluator accepts chess state."""
    board: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )
    state = ChessState(board=board)

    evaluator = create_game_board_evaluator_for_game_kind(
        game_kind=GameKind.CHESS,
        gui=False,
        can_oracle=False,
    )

    oracle, chi = evaluator.evaluate(state)

    assert oracle is None
    assert isinstance(chi, StateEvaluation)


def test_gui_evaluator_publishes_oracle_field() -> None:
    """Test gui evaluator publishes oracle field."""
    board: IBoard = create_board(
        use_rust_boards=False,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )
    state = ChessState(board=board)

    evaluator = create_game_board_evaluator_for_game_kind(
        game_kind=GameKind.CHESS,
        gui=True,
        can_oracle=False,
    )
    assert isinstance(evaluator, ObservableGameStateEvaluator)

    out: queue.Queue[GuiUpdate] = queue.Queue()
    scope = make_scope(session_id="session", match_id="match", game_id="game")
    publisher = GuiPublisher(
        out=out,
        schema_version=1,
        game_kind=GameKind.CHESS,
        scope=scope,
    )

    evaluator.subscribe(publisher)
    _ = evaluator.evaluate(state)

    update = out.get_nowait()
    assert isinstance(update.payload, UpdEvaluation)


if __name__ == "__main__":
    test_chess_evaluator_accepts_chess_state(False)
    test_chess_evaluator_accepts_chess_state(True)
    test_gui_evaluator_publishes_oracle_field()
    print("all tests passed")
