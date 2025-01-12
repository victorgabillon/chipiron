import chess
import pytest

from chipiron.environments.chess import BoardChi
from chipiron.environments.chess.board import (
    BoardModificationP,
    IBoard,
    create_board,
    create_board_chi,
    fen,
)
from chipiron.environments.chess.board.board_modification import (
    PieceInSquare,
    compute_modifications,
)
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.environments.chess.move import moveUci

examples: list[tuple[fen, moveUci, list[PieceInSquare], list[PieceInSquare]]] = [
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e2e3",
        [PieceInSquare(square=chess.E2, piece=chess.PAWN, color=chess.WHITE)],
        [PieceInSquare(square=chess.E3, piece=chess.PAWN, color=chess.WHITE)],
    ),
    (
        "8/5P2/8/3k4/7K/8/8/8 w - - 0 1",
        "f7f8b",
        [PieceInSquare(square=chess.F7, piece=chess.PAWN, color=chess.WHITE)],
        [PieceInSquare(square=chess.F8, piece=chess.BISHOP, color=chess.WHITE)],
    ),
    (
        "rnbqkb1r/pppp1ppp/5n2/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3",
        "d5e6",
        [
            PieceInSquare(square=chess.D5, piece=chess.PAWN, color=chess.WHITE),
            PieceInSquare(square=chess.E5, piece=chess.PAWN, color=chess.BLACK),
        ],
        [PieceInSquare(square=chess.E6, piece=chess.PAWN, color=chess.WHITE)],
    ),
    (
        "rnbqkbnr/ppp3pp/3ppp2/8/8/4PN2/PPPPBPPP/RNBQK2R w KQkq - 0 4",
        "e1g1",
        [
            PieceInSquare(square=chess.E1, piece=chess.KING, color=chess.WHITE),
            PieceInSquare(square=chess.H1, piece=chess.ROOK, color=chess.WHITE),
        ],
        [
            PieceInSquare(square=chess.G1, piece=chess.KING, color=chess.WHITE),
            PieceInSquare(square=chess.F1, piece=chess.ROOK, color=chess.WHITE),
        ],
    ),
    (
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "e4d5",
        [
            PieceInSquare(square=chess.E4, piece=chess.PAWN, color=chess.WHITE),
            PieceInSquare(square=chess.D5, piece=chess.PAWN, color=chess.BLACK),
        ],
        [
            PieceInSquare(square=chess.D5, piece=chess.PAWN, color=chess.WHITE),
        ],
    ),
]


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
def test_modifications(use_rust_boards: bool) -> None:
    fen_original: fen
    move_uci: moveUci
    removals: list[PieceInSquare]
    appearances: list[PieceInSquare]

    for fen_original, move_uci, removals, appearances in examples:
        board: IBoard = create_board(
            use_rust_boards=use_rust_boards,
            fen_with_history=FenPlusHistory(current_fen=fen_original),
            sort_legal_moves=True,
            use_board_modification=True,
        )

        board_modification: BoardModificationP | None = board.play_move_key(
            move=board.get_move_key_from_uci(move_uci=move_uci)
        )
        assert board_modification is not None

        assert set(board_modification.removals) == set(removals)
        assert set(board_modification.appearances) == set(appearances)


def test_compute_modifications() -> None:
    fen_original: fen
    move_uci: moveUci
    removals: list[PieceInSquare]
    appearances: list[PieceInSquare]

    for fen_original, move_uci, removals, appearances in examples:
        board_chi: BoardChi = create_board_chi(
            fen_with_history=FenPlusHistory(current_fen=fen_original),
            sort_legal_moves=True,
            use_board_modification=True,
        )

        previous_pawns = board_chi.chess_board.pawns
        previous_kings = board_chi.chess_board.kings
        previous_queens = board_chi.chess_board.queens
        previous_rooks = board_chi.chess_board.rooks
        previous_bishops = board_chi.chess_board.bishops
        previous_knights = board_chi.chess_board.knights
        previous_occupied_white = board_chi.chess_board.occupied_co[chess.WHITE]
        previous_occupied_black = board_chi.chess_board.occupied_co[chess.BLACK]

        board_chi.play_move_key(move=board_chi.get_move_key_from_uci(move_uci=move_uci))

        new_pawns = board_chi.chess_board.pawns
        new_kings = board_chi.chess_board.kings
        new_queens = board_chi.chess_board.queens
        new_rooks = board_chi.chess_board.rooks
        new_bishops = board_chi.chess_board.bishops
        new_knights = board_chi.chess_board.knights
        new_occupied_white = board_chi.chess_board.occupied_co[chess.WHITE]
        new_occupied_black = board_chi.chess_board.occupied_co[chess.BLACK]

        board_modifications_2 = compute_modifications(
            previous_bishops=previous_bishops,
            previous_pawns=previous_pawns,
            previous_kings=previous_kings,
            previous_knights=previous_knights,
            previous_queens=previous_queens,
            previous_occupied_white=previous_occupied_white,
            previous_rooks=previous_rooks,
            previous_occupied_black=previous_occupied_black,
            new_kings=new_kings,
            new_bishops=new_bishops,
            new_pawns=new_pawns,
            new_queens=new_queens,
            new_rooks=new_rooks,
            new_knights=new_knights,
            new_occupied_black=new_occupied_black,
            new_occupied_white=new_occupied_white,
        )

        assert set(board_modifications_2.removals) == set(removals)
        assert set(board_modifications_2.appearances) == set(appearances)


if __name__ == "__main__":

    test_compute_modifications()
    use_rusty_board: bool
    for use_rusty_board in [True, False]:
        test_modifications(use_rust_boards=use_rusty_board)
