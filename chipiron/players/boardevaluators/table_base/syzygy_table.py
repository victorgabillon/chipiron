"""
Module for the SyzygyTable class.
"""

from typing import Protocol

import chess.syzygy

from chipiron.environments.chess.board import IBoard
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.over_event import HowOver, OverTags, Winner


class SyzygyTable[T_Board: IBoard](Protocol):
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

    def fast_in_table(self, board: T_Board) -> bool:
        """
        Check if the given board is suitable for fast tablebase lookup.

        Args:
            board (boards.BoardChi): The board to check.

        Returns:
            bool: True if the board is suitable for fast lookup, False otherwise.
        """
        return board.number_of_pieces_on_the_board() < 6

    def in_table(self, board: T_Board) -> bool:
        """
        Check if the given board is in the tablebase.

        Args:
            board (boards.BoardChi): The board to check.

        Returns:
            bool: True if the board is in the tablebase, False otherwise.
        """
        try:
            self.wdl(board=board)
        except KeyError:
            return False
        return True

    def wdl(self, board: T_Board) -> int: ...

    def get_over_event(self, board: T_Board) -> tuple[Winner, HowOver]:
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
                who_is_winner_ = (
                    Winner.WHITE if board.turn == chess.WHITE else Winner.BLACK
                )
            if val < 0:
                who_is_winner_ = (
                    Winner.WHITE if board.turn == chess.BLACK else Winner.BLACK
                )
        else:
            how_over_ = HowOver.DRAW

        return who_is_winner_, how_over_

    def val(self, board: T_Board) -> int:
        """
        Get the value of the given board from the tablebase.

        Args:
            board (boards.BoardChi): The board to get the value for.

        Returns:
            int: The value of the board from the tablebase.
        """
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val: int = self.wdl(board)
        return val

    def value_white(self, board: T_Board) -> int:
        """
        Get the value of the given board for the white player.

        Args:
            board (boards.BoardChi): The board to get the value for.

        Returns:
            int: The value of the board for the white player.
        """
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val: int = self.wdl(board)
        if board.turn == chess.WHITE:
            return val * 100000
        else:
            return val * -10000

    def get_over_tag(self, board: T_Board) -> OverTags:
        """
        Get the over tag for the given board.

        Args:
            board (boards.BoardChi): The board to get the over tag for.

        Returns:
            OverTags: The over tag for the board.
        """
        val = self.wdl(board)
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

    def string_result(self, board: T_Board) -> str:
        """
        Get the string representation of the result for the given board.

        Args:
            board (boards.BoardChi): The board to get the result for.

        Returns:
            str: The string representation of the result.
        """
        val = self.wdl(board)
        player_to_move = "white" if board.turn == chess.WHITE else "black"
        if val > 0:
            return "WIN for player " + player_to_move
        elif val == 0:
            return "DRAW"
        else:
            return "LOSS for player " + player_to_move

    def dtz(self, board: T_Board) -> int:
        """
        Get the distance-to-zero (DTZ) value for the given board.

        Args:
            board (boards.BoardChi): The board to get the DTZ value for.

        Returns:
            int: The DTZ value for the board.
        """
        ...

    def best_move(self, board: T_Board) -> moveKey:
        """
        Get the best move according to the tablebase for the given board.

        Args:
            board (boards.BoardChi): The board to find the best move for.

        Returns:
            chess.Move: The best move according to the tablebase.
        """
        all_moves: list[moveKey] = board.legal_moves_.get_all()
        # avoid draws by 50 move rules in winning position, # otherwise look
        # for it to make it last and preserve pieces in case of mistake by opponent

        best_value = -1000000000000000000000

        assert all_moves
        best_move: moveKey = all_moves[0]
        for move in all_moves:
            board_copy: T_Board = board.copy(stack=True)
            board_copy.play_move_key(move=move)
            val_player_next_board = self.val(board_copy)
            val_player_node = -val_player_next_board
            dtz_player_next_board = self.dtz(board_copy)
            dtz_player_node = -dtz_player_next_board
            if val_player_node > 0:  # winning position
                new_value = board.is_zeroing(move) * 100 - dtz_player_node + 1000
            elif val_player_node == 0:
                new_value = -board.is_zeroing(move) * 100 + dtz_player_node
            elif val_player_node < 0:
                new_value = -board.is_zeroing(move) * 100 + dtz_player_node - 1000

            if new_value > best_value:
                best_value = new_value
                best_move = move
        return best_move
