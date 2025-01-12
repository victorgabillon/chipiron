"""
Module for the SyzygyTable class.
"""

import shakmaty_python_binding

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.board import RustyBoardChi
from chipiron.utils import path

from .syzygy_table import SyzygyTable


class SyzygyRustTable(SyzygyTable[RustyBoardChi]):
    """
    A class representing a Syzygy tablebase for chess endgame analysis.

    Attributes:
        table_base (chess.syzygy.Tablebase): The Syzygy tablebase object.

    Methods:
        fast_in_table(board: boards.BoardChi) -> bool:
            Check if the given board is suitable for fast tablebase lookup.

        in_table(board: boards.BoardChi) -> bool:
            Check if the given board is in the tablebase.

        get_over_event(board: boards.BoardChi) -> tuple[Winner, HowOver]:
            Get the winner and how the game is over for the given board.

        val(board: boards.BoardChi) -> int:
            Get the value of the given board from the tablebase.

        value_white(board: boards.BoardChi) -> int:
            Get the value of the given board for the white player.

        get_over_tag(board: boards.BoardChi) -> OverTags:
            Get the over tag for the given board.

        string_result(board: boards.BoardChi) -> str:
            Get the string representation of the result for the given board.

        dtz(board: boards.BoardChi) -> int:
            Get the distance-to-zero (DTZ) value for the given board.

        best_move(board: boards.BoardChi) -> chess.Move:
            Get the best move according to the tablebase for the given board.
    """

    table_base: shakmaty_python_binding.MyTableBase

    def __init__(self, path_to_table: path):
        """
        Initialize the SyzygyTable object.

        Args:
            path_to_table (path): The path to the Syzygy tablebase.
        """
        path_to_table_str: str = str(path_to_table)
        self.table_base = shakmaty_python_binding.MyTableBase(path_to_table_str)

    def wdl(self, board: boards.RustyBoardChi) -> int:
        wdl: int = self.table_base.probe_wdl(board.chess_)
        return wdl

    def dtz(self, board: boards.RustyBoardChi) -> int:
        """
        Get the distance-to-zero (DTZ) value for the given board.

        Args:
            board (boards.BoardChi): The board to get the DTZ value for.

        Returns:
            int: The DTZ value for the board.
        """
        dtz: int = self.table_base.probe_dtz(board.chess_)
        return dtz
