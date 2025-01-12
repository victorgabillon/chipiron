import chess
import pytest

from chipiron.environments.chess.board import IBoard, create_board, fen
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.environments.chess.move import moveUci


@pytest.mark.parametrize(("use_rusty_board"), (True, False))
def test_copy(use_rusty_board: bool) -> None:
    board: IBoard = create_board(
        use_rust_boards=use_rusty_board,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
        sort_legal_moves=True,
    )

    copy_board: IBoard = board.copy(stack=True, deep_copy_legal_moves=True)

    all_moves_keys_chi = board.legal_moves.get_all()

    board.play_move_key(move=all_moves_keys_chi[0])

    assert board.fen == board.legal_moves.fen
    assert copy_board.fen == copy_board.legal_moves.fen
    assert copy_board.fen != board.fen


@pytest.mark.parametrize(("use_rusty_board"), (True, False))
def test_move(use_rusty_board: bool) -> None:
    examples: list[tuple[fen, moveUci, fen]] = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e3",
            "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ),
        ("8/5P2/8/3k4/7K/8/8/8 w - - 0 1", "f7f8b", "5B2/8/8/3k4/7K/8/8/8 b - - 0 1"),
        (
            "rnbqkb1r/pppp1ppp/5n2/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3",
            "d5e6",
            "rnbqkb1r/pppp1ppp/4Pn2/8/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 3",
        ),
        (
            "rnbqkbnr/ppp3pp/3ppp2/8/8/4PN2/PPPPBPPP/RNBQK2R w KQkq - 0 4",
            "e1g1",
            "rnbqkbnr/ppp3pp/3ppp2/8/8/4PN2/PPPPBPPP/RNBQ1RK1 b kq - 1 4",
        ),
        (
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "e4d5",
            "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        ),
    ]

    fen_original: fen
    move_uci: moveUci
    fen_next: fen

    for fen_original, move_uci, fen_next in examples:
        board: IBoard = create_board(
            use_rust_boards=use_rusty_board,
            fen_with_history=FenPlusHistory(current_fen=fen_original),
            sort_legal_moves=True,
        )

        board.play_move_key(move=board.get_move_key_from_uci(move_uci=move_uci))

        assert board.fen == fen_next


if __name__ == "__main__":
    use_rusty_board: bool
    for use_rusty_board in [True, False]:
        test_copy(use_rusty_board=use_rusty_board)
        test_move(use_rusty_board=use_rusty_board)
