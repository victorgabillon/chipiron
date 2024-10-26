"""
Module for the SyzygyTable class.
"""
import chess.syzygy
import shakmaty_python_binding

import chipiron.environments.chess.board as boards
from chipiron.players.boardevaluators.over_event import Winner, HowOver, OverTags
from chipiron.utils import path
from .syzygy_table import SyzygyTable


class SyzygyRustTable(SyzygyTable):
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

    def __init__(
            self,
            path_to_table: path
    ):
        """
        Initialize the SyzygyTable object.

        Args:
            path_to_table (path): The path to the Syzygy tablebase.
        """
        path_to_table_str: str = str(path_to_table)
        self.table_base = shakmaty_python_binding.MyTableBase(path_to_table_str)

    def fast_in_table(
            self,
            board: boards.RustyBoardChi
    ) -> bool:
        """
        Check if the given board is suitable for fast tablebase lookup.

        Args:
            board (boards.BoardChi): The board to check.

        Returns:
            bool: True if the board is suitable for fast lookup, False otherwise.
        """
        return board.number_of_pieces_on_the_board() < 6

    def in_table(
            self,
            board: boards.RustyBoardChi
    ) -> bool:
        """
        Check if the given board is in the tablebase.

        Args:
            board (boards.BoardChi): The board to check.

        Returns:
            bool: True if the board is in the tablebase, False otherwise.
        """
        try:
            self.table_base.probe_wdl(board.chess_)
        except KeyError:
            return False
        return True

    def get_over_event(
            self,
            board: boards.RustyBoardChi
    ) -> tuple[Winner, HowOver]:
        """
        Get the winner and how the game is over for the given board.

        Args:
            board (boards.BoardChi): The board to analyze.

        Returns:
            tuple[Winner, HowOver]: The winner and how the game is over.
        """
        val: int = self.val(board)

        who_is_winner_: Winner = Winner.NO_KNOWN_WINNER
        how_over_: HowOver
        if val != 0:
            how_over_ = HowOver.WIN
            if val > 0:
                who_is_winner_ = Winner.WHITE if board.turn == chess.WHITE else Winner.BLACK
            if val < 0:
                who_is_winner_ = Winner.WHITE if board.turn == chess.BLACK else Winner.BLACK
        else:
            how_over_ = HowOver.DRAW

        return who_is_winner_, how_over_

    def val(
            self,
            board: boards.RustyBoardChi
    ) -> int:
        """
        Get the value of the given board from the tablebase.

        Args:
            board (boards.BoardChi): The board to get the value for.

        Returns:
            int: The value of the board from the tablebase.
        """
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val: int = self.table_base.probe_wdl(board.chess_)
        return val

    def value_white(
            self,
            board: boards.RustyBoardChi
    ) -> int:
        """
        Get the value of the given board for the white player.

        Args:
            board (boards.BoardChi): The board to get the value for.

        Returns:
            int: The value of the board for the white player.
        """
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val: int = self.table_base.probe_wdl(board.chess_)
        if board.turn == chess.WHITE:
            return val * 100000
        else:
            return val * -10000

    def get_over_tag(
            self,
            board: boards.RustyBoardChi
    ) -> OverTags:
        """
        Get the over tag for the given board.

        Args:
            board (boards.BoardChi): The board to get the over tag for.

        Returns:
            OverTags: The over tag for the board.
        """
        val = self.table_base.probe_wdl(board.chess_)
        if val > 0:
            if board.turn == chess.WHITE:
                return OverTags.TAG_WIN_WHITE
            else:
                return OverTags.TAG_WIN_BLACK
        elif val == 0:
            return OverTags.TAG_DRAW
        else:
            if board.turn == chess.WHITE:
                return OverTags.TAG_WIN_BLACK
            else:
                return OverTags.TAG_WIN_WHITE

    def string_result(
            self,
            board: boards.RustyBoardChi
    ) -> str:
        """
        Get the string representation of the result for the given board.

        Args:
            board (boards.BoardChi): The board to get the result for.

        Returns:
            str: The string representation of the result.
        """
        val = self.table_base.probe_wdl(board.chess_)
        player_to_move = 'white' if board.turn == chess.WHITE else 'black'
        if val > 0:
            return 'WIN for player ' + player_to_move
        elif val == 0:
            return 'DRAW'
        else:
            return 'LOSS for player ' + player_to_move

    def dtz(
            self,
            board: boards.RustyBoardChi
    ) -> int:
        """
        Get the distance-to-zero (DTZ) value for the given board.

        Args:
            board (boards.BoardChi): The board to get the DTZ value for.

        Returns:
            int: The DTZ value for the board.
        """
        dtz: int = self.table_base.probe_dtz(board.chess_)
        return dtz
