"""Module for test game."""

import chess
import pytest
from atomheart import ChessDynamics
from atomheart.games.chess.board import IBoard, create_board
from atomheart.games.chess.board.utils import FenPlusHistory

from chipiron.environments.chess.types import ChessState
from chipiron.games.game import Game
from chipiron.games.game.game_playing_status import GamePlayingStatus


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
def test_game_rewind(use_rust_boards: bool) -> None:
    """Test game rewind."""
    board: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )

    state = ChessState(board=board)

    game_playing_status: GamePlayingStatus = GamePlayingStatus()

    dynamics = ChessDynamics()
    game: Game[ChessState] = Game(
        state=state, dynamics=dynamics, playing_status=game_playing_status
    )

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(action=dynamics.action_from_name(game.state, "e2e4"))

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(action=dynamics.action_from_name(game.state, "e2e4"))

    game.playing_status.play()
    game.play_move(action=dynamics.action_from_name(game.state, "g8f6"))

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(action=dynamics.action_from_name(game.state, "g8f6"))


if __name__ == "__main__":
    test_game_rewind(use_rust_boards=False)
    test_game_rewind(use_rust_boards=True)
