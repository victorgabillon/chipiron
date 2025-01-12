import chess
import pytest

from chipiron.environments.chess.board import IBoard, create_board
from chipiron.environments.chess.board.utils import FenPlusHistory

from .game import Game
from .game_playing_status import GamePlayingStatus


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
def test_game_rewind(use_rust_boards: bool) -> None:
    board: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )

    game_playing_status: GamePlayingStatus = GamePlayingStatus()

    game: Game = Game(board=board, playing_status=game_playing_status)

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(move=game.board.get_move_key_from_uci("e2e4"))

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(move=game.board.get_move_key_from_uci("e2e4"))

    game.playing_status.play()
    game.play_move(move=game.board.get_move_key_from_uci("g8f6"))

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(move=game.board.get_move_key_from_uci("g8f6"))


if __name__ == "__main__":
    test_game_rewind(use_rust_boards=False)
    test_game_rewind(use_rust_boards=True)
