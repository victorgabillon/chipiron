"""Tests for the Syzygy table-based board evaluator."""

from typing import TYPE_CHECKING

from atomheart.board.factory import (
    create_board_chi,
    create_rust_board,
)
from atomheart.board.utils import (
    FenPlusHistory,
)

from chipiron.players.boardevaluators.table_base.factory import (
    create_syzygy_python,
    create_syzygy_rust,
)

if TYPE_CHECKING:
    from atomheart.board.board_chi import BoardChi
    from atomheart.board.rusty_board import RustyBoardChi
    from atomheart.move.imove import MoveKey

    from chipiron.players.boardevaluators.table_base import SyzygyTable


def test_best_move_syzygy_table() -> None:
    """Test the best move selection from the Syzygy table."""

    syzygy_table_chi: SyzygyTable[BoardChi] | None
    syzygy_table_chi = create_syzygy_python()

    if syzygy_table_chi is None:
        print(
            "Skipping test as no syzygy table found (to avoid some ci issues where the table is not available)"
        )
        return
    board_chi: BoardChi
    board_chi = create_board_chi(
        fen_with_history=FenPlusHistory(
            current_fen="6k1/p7/8/8/7N/7K/2N5/8 w - - 0 1",
        )
    )
    best_move_chi: MoveKey = syzygy_table_chi.best_move(board_chi)

    assert "h3g4" == board_chi.get_uci_from_move_key(best_move_chi)

    syzygy_table: SyzygyTable[RustyBoardChi] | None
    syzygy_table = create_syzygy_rust()
    assert syzygy_table is not None

    board_rust: RustyBoardChi
    board_rust = create_rust_board(
        fen_with_history=FenPlusHistory(
            current_fen="6k1/p7/8/8/7N/7K/2N5/8 w - - 0 1",
        )
    )
    best_move_rust: MoveKey = syzygy_table.best_move(board_rust)

    assert "h3g4" == board_rust.get_uci_from_move_key(best_move_rust)


if __name__ == "__main__":
    test_best_move_syzygy_table()
    print("all tests passed")
