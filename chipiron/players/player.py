"""
Module for the Player class.
"""

from typing import TYPE_CHECKING

from atomheart.board import BoardFactory, IBoard
from atomheart.board.utils import FenPlusHistory

from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable
from valanga.policy import Recommendation, BranchSelector
from valanga.game import Seed
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from atomheart.move import MoveUci
    from atomheart.move.imove import MoveKey

PlayerId = str


class Player:
    """
    Player selects moves on a given board
    """

    #  difference between player and treebuilder includes the fact
    #  that now a player can be a mixture of multiple decision rules
    id: PlayerId
    main_move_selector: BranchSelector
    syzygy: AnySyzygyTable | None
    board_factory: BoardFactory

    def __init__(
        self,
        name: str,
        syzygy: AnySyzygyTable | None,
        main_move_selector: BranchSelector,
        board_factory: BoardFactory,
    ):
        self.id = name
        self.main_move_selector: BranchSelector = main_move_selector
        self.syzygy_player = syzygy
        self.board_factory = board_factory

    def get_id(self) -> PlayerId:
        """
        Returns the ID of the player.

        Returns:
            PlayerId: The player's ID.
        """
        return self.id

    def select_move(
        self, fen_plus_history: FenPlusHistory, seed_int: Seed
    ) -> Recommendation:
        """
        Returns the best move computed by the player.
        The player has the option to ask the syzygy table to play it.

        Args:
            board (BoardChi): The current board state.
            seed_int (seed): The seed for move selection.

        Returns:
            Recommendation: The recommended move.
        """

        board: IBoard = self.board_factory(fen_with_history=fen_plus_history)

        move_recommendation: Recommendation
        # if there is only one possible legal move in the position, do not think, choose it.
        if not board.legal_moves:
            raise ValueError("No legal moves in this position")
        if board.legal_moves.more_than_one_move() or self.id != "Human":
            # if len(list(board.legal_moves))>1 or self.id != 'Human':

            # if the play with syzygy option is on test if the position is in the database to play syzygy
            if self.syzygy_player is not None and self.syzygy_player.fast_in_table(
                board
            ):
                chipiron_logger.info("Playing with Syzygy")
                best_move_key: MoveKey = self.syzygy_player.best_move(board)
                best_move_uci: MoveUci = board.get_uci_from_move_key(best_move_key)
                move_recommendation = Recommendation(move=best_move_uci)
            else:
                chipiron_logger.info(
                    "Playing with player (not Syzygy) %s\n%s", self.id, board
                )
                move_recommendation = self.main_move_selector.select_move(
                    board=board, move_seed=seed_int
                )
        else:
            move_key: MoveKey = list(board.legal_moves)[0]
            move_uci: MoveUci = board.get_uci_from_move_key(move_key)
            move_recommendation = Recommendation(move=move_uci)

        return move_recommendation
