from chipiron.environments.chess import BoardChi
from chipiron.environments.chess.board import create_board_chi, fen
from chipiron.environments.chess.board.factory import create_rust_board
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.environments.chess.move import moveUci


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

    assert (board_chi.fen == board_chi.legal_moves.chess_board.fen())
    assert (copy_board.fen == copy_board.legal_moves.chess_board.fen())
    assert (copy_board.fen != board_chi.fen)


def test_move() -> None:
    examples: list[tuple[fen, moveUci, fen]] = [
        (
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'e2e3',
            'rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
        ),
        (
            '8/5P2/8/3k4/7K/8/8/8 w - - 0 1',
            'f7f8b',
            '5B2/8/8/3k4/7K/8/8/8 b - - 0 1'
        ),
        (
            'rnbqkb1r/pppp1ppp/5n2/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3',
            'd5e6',
            'rnbqkb1r/pppp1ppp/4Pn2/8/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 3'
        )
    ]

    fen_original: fen
    move_uci: moveUci
    fen_next: fen
    for fen_original, move_uci, fen_next in examples:
        board_chi = create_board_chi(
            fen_with_history=FenPlusHistory(
                current_fen=fen_original
            ),
            sort_legal_moves=True

        )

        board_rust = create_rust_board(
            fen_with_history=FenPlusHistory(
                current_fen=fen_original
            ),
            sort_legal_moves=True
        )

        board_chi.play_move_key(move=board_chi.get_move_key_from_uci(move_uci=move_uci))
        board_rust.play_move_key(move=board_rust.get_move_key_from_uci(move_uci=move_uci))

        assert (board_chi.fen == fen_next)
        assert (board_rust.fen == fen_next)


if __name__ == '__main__':
    test_copy()
    test_move()
