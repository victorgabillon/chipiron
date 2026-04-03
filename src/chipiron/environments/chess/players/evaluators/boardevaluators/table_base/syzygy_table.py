"""Module for the SyzygyTable class."""

from typing import TYPE_CHECKING, Protocol

from atomheart.games.chess.board import IBoard
from atomheart.games.chess.move.imove import MoveKey
from valanga import Color, Outcome, OverEvent
from valanga.evaluations import Certainty, Value

if TYPE_CHECKING:
    from collections.abc import Sequence


class SyzygyTable[BoardT: IBoard](Protocol):
    """A class representing a Syzygy tablebase for chess endgame analysis.

    Attributes:
        table_base (chess.syzygy.Tablebase): The Syzygy tablebase object.

    Methods:
        fast_in_table(board: T_Board) -> bool:
            Check if the given board is suitable for fast tablebase lookup.

        in_table(board: T_Board) -> bool:
            Check if the given board is in the tablebase.

        get_over_event(board: T_Board) -> tuple[Winner, HowOver]:
            Get the winner and how the game is over for the given board.

        val(board: T_Board) -> int:
            Get the value of the given board from the tablebase.

        value_white(board: T_Board) -> int:
            Get the value of the given board for the white player.

        string_result(board: T_Board) -> str:
            Get the string representation of the result for the given board.

        dtz(board: T_Board) -> int:
            Get the distance-to-zero (DTZ) value for the given board.

        best_move(board: T_Board) -> MoveKey:
            Get the best move according to the tablebase for the given board.

    """

    def fast_in_table(self, board: BoardT) -> bool:
        """Check if the given board is suitable for fast tablebase lookup.

        Args:
            board (boards.BoardChi): The board to check.

        Returns:
            bool: True if the board is suitable for fast lookup, False otherwise.

        """
        return board.number_of_pieces_on_the_board() < 6

    def in_table(self, board: BoardT) -> bool:
        """Check if the given board is in the tablebase.

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

    def wdl(self, board: BoardT) -> int:
        """Probe the tablebase for the win/draw/loss value of the given board.

        Args:
            board (T_Board): The board to probe.

        Returns:
            int: 2 if the side to move is winning, 0 if draw, -2 if losing.

        """
        # Example implementation, replace with actual tablebase probing logic
        # For now, raise NotImplementedError to indicate it must be implemented
        raise NotImplementedError("wdl method must be implemented in subclass")

    def get_over_event(self, board: BoardT) -> tuple[Color | None, Outcome]:
        """Get the winner and how the game is over for the given board.

        Args:
            board (boards.BoardChi): The board to analyze.

        Returns:
            tuple[Winner, HowOver]: The winner and how the game is over.

        """
        val: int = self.val(board)

        who_is_winner_: Color | None = None
        how_over_: Outcome
        if val != 0:
            how_over_ = Outcome.WIN
            if val > 0:
                who_is_winner_ = board.turn  # The player to move is winning
            if val < 0:
                who_is_winner_ = (
                    Color.WHITE if board.turn == Color.BLACK else Color.BLACK
                )
        else:
            how_over_ = Outcome.DRAW

        return who_is_winner_, how_over_

    def val(self, board: BoardT) -> int:
        """Get the value of the given board from the tablebase.

        Args:
            board (boards.BoardChi): The board to get the value for.

        Returns:
            int: The value of the board from the tablebase.

        """
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val: int = self.wdl(board)
        return val

    def evaluate(self, board: BoardT) -> Value:
        """Evaluate the given board using the tablebase.

        Args:
            board (boards.BoardChi): The board to evaluate.

        Returns:
            Value: The evaluation of the board, including score and certainty.

        """
        val: int = self.val(board)
        return Value(
            score=val,
            certainty=Certainty.FORCED,
            over_event=OverEvent(
                outcome=Outcome.WIN if val != 0 else Outcome.DRAW,
                winner=Color.WHITE if val > 0 else (Color.BLACK if val < 0 else None),
                termination=None,
            ),
        )

    def value_white(self, state: BoardT) -> int:
        """Get the value of the given board for the white player.

        Args:
            state (boards.BoardChi): The board to get the value for.

        Returns:
            int: The value of the board for the white player.

        """
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val: int = self.wdl(state)
        if state.turn == Color.WHITE:
            return val * 100000
        return val * -10000

    def string_result(self, board: BoardT) -> str:
        """Get the string representation of the result for the given board.

        Args:
            board (boards.BoardChi): The board to get the result for.

        Returns:
            str: The string representation of the result.

        """
        val = self.wdl(board)
        player_to_move = "white" if board.turn == Color.WHITE else "black"
        if val > 0:
            return "WIN for player " + player_to_move
        if val == 0:
            return "DRAW"
        return "LOSS for player " + player_to_move

    def dtz(self, board: BoardT) -> int:
        """Get the distance-to-zero (DTZ) value for the given board.

        Args:
            board (boards.BoardChi): The board to get the DTZ value for.

        Returns:
            int: The DTZ value for the board.

        """
        # Example implementation, replace with actual tablebase probing logic
        # For now, return a default value (e.g., 0)
        raise NotImplementedError("wdl method must be implemented in subclass")

    def best_move(self, board: BoardT) -> MoveKey:
        """Get the best move according to the tablebase for the given board.

        Args:
            board (boards.BoardChi): The board to find the best move for.

        Returns:
            chess.Move: The best move according to the tablebase.

        """
        all_moves: Sequence[MoveKey] = board.legal_moves.get_all()
        # avoid draws by 50 move rules in winning position, # otherwise look
        # for it to make it last and preserve pieces in case of mistake by opponent

        best_value = -1000000000000000000000

        assert all_moves
        best_move: MoveKey = all_moves[0]
        move: MoveKey
        for move in all_moves:
            board_copy: BoardT = board.copy(stack=True)
            board_copy.play_move_key(move=move)
            val_player_next_board = self.val(board_copy)
            val_player_node = -val_player_next_board
            dtz_player_next_board = self.dtz(board_copy)
            dtz_player_node = -dtz_player_next_board
            if val_player_node > 0:  # winning position
                new_value = (
                    board.is_zeroing(move) * 100
                    - dtz_player_node * (1 - int(board.is_zeroing(move)))
                    + 10000
                )
            elif val_player_node == 0:  # drawing position
                new_value = -board.is_zeroing(move) * 100 + dtz_player_node * (
                    1 - int(board.is_zeroing(move))
                )
            else:  # losing position
                new_value = (
                    -board.is_zeroing(move) * 100
                    - dtz_player_node * (1 - int(board.is_zeroing(move)))
                    - 10000
                )

            if new_value > best_value:
                best_value = new_value
                best_move = move

        return best_move
