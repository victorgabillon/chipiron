import chess
import pytest
from atomheart.board import IBoard, create_board
from atomheart.board.utils import FenPlusHistory

from .game import Game
from .game_playing_status import GamePlayingStatus
from atomheart.board.valanga_adapter import ValangaChessState

@pytest.mark.parametrize(("use_rust_boards"), (True, False))
def test_game_rewind(use_rust_boards: bool) -> None:
    board: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )

    state = ValangaChessState(board=board)

    game_playing_status: GamePlayingStatus = GamePlayingStatus()

    game: Game = Game(state=state, playing_status=game_playing_status)

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(action=game.state.branch_key_from_name("e2e4"))

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(action=game.state.branch_key_from_name("e2e4"))

    game.playing_status.play()
    game.play_move(action=game.state.branch_key_from_name("g8f6"))

    game.playing_status.pause()
    game.rewind_one_move()

    game.playing_status.play()
    game.play_move(action=game.state.branch_key_from_name("g8f6"))


if __name__ == "__main__":
    test_game_rewind(use_rust_boards=False)
    test_game_rewind(use_rust_boards=True)
