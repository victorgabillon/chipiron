from chipiron.environments.chess import BoardChi
from chipiron.environments.chess.board import create_board_chi
from chipiron.environments.chess.board.utils import FenPlusHistory


def test_copy() -> None:
    board_chi: BoardChi = create_board_chi(
        fen_with_history=FenPlusHistory(
            current_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        ),
        sort_legal_moves=True

    )

    copy_board: BoardChi = board_chi.copy(
        stack=True,
        deep_copy_legal_moves=True
    )

    all_moves_keys_chi = board_chi.legal_moves.get_all()

    board_chi.play_move_key(move=all_moves_keys_chi[0])

    print('fens', board_chi.fen, board_chi.legal_moves.chess_board.fen())
    assert (board_chi.fen == board_chi.legal_moves.chess_board.fen())

    print('copy fens', copy_board.fen, copy_board.legal_moves.chess_board.fen())
    assert (copy_board.fen == copy_board.legal_moves.chess_board.fen())

    assert (copy_board.fen != board_chi.fen)


if __name__ == '__main__':
    test_copy()
