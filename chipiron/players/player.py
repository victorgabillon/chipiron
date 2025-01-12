"""
Module for the Player class.
"""

from typing import Any

from chipiron.environments.chess.board import IBoard
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.utils import seed

from .move_selector.move_selector import MoveRecommendation, MoveSelector

playerId = str


class Player:
    """
    Player selects moves on a given board
    """

    #  difference between player and treebuilder includes the fact
    #  that now a player can be a mixture of multiple decision rules
    id: playerId
    main_move_selector: MoveSelector

    def __init__(
        self,
        name: str,
        syzygy: SyzygyTable[Any] | None,
        main_move_selector: MoveSelector,
    ):
        self.id = name
        self.main_move_selector: MoveSelector = main_move_selector
        self.syzygy_player = syzygy

    def select_move(self, board: IBoard, seed_int: seed) -> MoveRecommendation:
        """
        Returns the best move computed by the player.
        The player has the option to ask the syzygy table to play it.

        Args:
            board (BoardChi): The current board state.
            seed_int (seed): The seed for move selection.

        Returns:
            MoveRecommendation: The recommended move.
        """

        move_recommendation: MoveRecommendation
        # if there is only one possible legal move in the position, do not think, choose it.
        if not board.legal_moves:
            raise Exception("No legal moves in this position")
        if board.legal_moves.more_than_one_move() or self.id != "Human":
            # if len(list(board.legal_moves))>1 or self.id != 'Human':

            # if the play with syzygy option is on test if the position is in the database to play syzygy
            if self.syzygy_player is not None and self.syzygy_player.fast_in_table(
                board
            ):
                print("Playing with Syzygy")
                best_move: moveKey = self.syzygy_player.best_move(board)
                move_recommendation = MoveRecommendation(move=best_move)

            else:
                print(f"Playing with player (not Syzygy) {self.id}\n{board}")
                move_recommendation = self.main_move_selector.select_move(
                    board=board, move_seed=seed_int
                )
        else:
            move_recommendation = MoveRecommendation(move=list(board.legal_moves)[0])

        return move_recommendation
