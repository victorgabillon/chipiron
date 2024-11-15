"""
Module for the Player class.
"""

from typing import Any

from chipiron.environments.chess.board import IBoard
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.utils import seed
from .move_selector.move_selector import MoveSelector, MoveRecommendation


class Player:
    """
    Player selects moves on a given board
    """
    #  difference between player and treebuilder includes the fact
    #  that now a player can be a mixture of multiple decision rules
    id: str
    main_move_selector: MoveSelector

    def __init__(
            self,
            name: str,
            syzygy: SyzygyTable[Any] | None,
            main_move_selector: MoveSelector):
        self.id = name
        self.main_move_selector: MoveSelector = main_move_selector
        self.syzygy_player = syzygy

    def select_move(
            self,
            board: IBoard[Any],
            seed_int: seed
    ) -> MoveRecommendation:
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
        all_legal_moves = list(board.legal_moves)
        if not all_legal_moves:
            raise Exception('No legal moves in this position')
        if len(all_legal_moves) == 1 and self.id != 'Human':
            move_recommendation = MoveRecommendation(move=all_legal_moves[0].uci())
        else:
            # if the play with syzygy option is on test if the position is in the database to play syzygy
            if self.syzygy_player is not None and self.syzygy_player.fast_in_table(board):
                print('Playing with Syzygy')
                best_move = self.syzygy_player.best_move(board)
                move_recommendation = MoveRecommendation(move=best_move.uci())

            else:
                print(f'Playing with player (not Syzygy) {self.id}\n{board}')
                move_recommendation = self.main_move_selector.select_move(
                    board=board,
                    move_seed=seed_int
                )

        return move_recommendation
