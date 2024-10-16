import chess

from chipiron.environments.chess.board import create_board, BoardChi
from chipiron.environments.chess.board.factory import create_rust_board
from chipiron.environments.chess.board.rusty_board import RustyBoardChi
from chipiron.environments.chess.board.utils import FenPlusMoveHistory


def test_three_fold_repetition_board_chi():
    # todo maybe this sis more a unit test for the is_game_over method atm
    board: BoardChi = create_board(
        fen_with_history=FenPlusMoveHistory(
            current_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    )

    board.play_move(chess.Move.from_uci('b1c3'))
    board.play_move(chess.Move.from_uci('b8c6'))
    board.play_move(chess.Move.from_uci('c3b1'))
    board.play_move(chess.Move.from_uci('c6b8'))

    board.play_move(chess.Move.from_uci('b1c3'))
    board.play_move(chess.Move.from_uci('b8c6'))
    board.play_move(chess.Move.from_uci('c3b1'))
    board.play_move(chess.Move.from_uci('c6b8'))

    assert(board.is_game_over())


def test_three_fold_repetition_rusty_board():
    # todo maybe this sis more a unit test for the is_game_over method atm
    board: RustyBoardChi = create_rust_board(
        fen_with_history=FenPlusMoveHistory(
            current_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    )

    board.play_move(chess.Move.from_uci('b1c3'))
    board.play_move(chess.Move.from_uci('b8c6'))
    board.play_move(chess.Move.from_uci('c3b1'))
    board.play_move(chess.Move.from_uci('c6b8'))

    board.play_move(chess.Move.from_uci('b1c3'))
    board.play_move(chess.Move.from_uci('b8c6'))
    board.play_move(chess.Move.from_uci('c3b1'))
    board.play_move(chess.Move.from_uci('c6b8'))

    assert(board.is_game_over())


test_three_fold_repetition_board_chi()
test_three_fold_repetition_rusty_board()
