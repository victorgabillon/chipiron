from .move_selector.move_selector import MoveSelector, MoveRecommendation
from chipiron.environments.chess.board.board import BoardChi


class Player:
    #  difference between player and treebuilder includes the fact
    #  that now a player can be a mixture of multiple decision rules

    id: str

    def __init__(
            self,
            name: str,
            syzygy_play,
            syzygy,
            main_move_selector: MoveSelector):
        self.id = name
        self.main_move_selector: MoveSelector = main_move_selector

        self.syzygy_player = None
        self.syzygy_play = syzygy_play
        if self.syzygy_play:
            self.syzygy_player = syzygy

    def select_move(
            self,
            board: BoardChi
    ) -> MoveRecommendation:
        """ returns the best move computed by the player.
        The player has the option to ask the syzygy table to play it"""

        move_recommendation: MoveRecommendation
        # if there is only one possible legal move in the position, do not think, choose it.
        all_legal_moves = list(board.legal_moves)
        if len(all_legal_moves) == 1 and self.id != 'Human':
            move_recommendation = MoveRecommendation(move=all_legal_moves[0])
        else:
            # if the play with syzygy option is on test if the position is in the database to play syzygy
            if self.syzygy_play and self.syzygy_player.fast_in_table(board):
                print('Playing with Syzygy')
                best_move = self.syzygy_player.best_move(board)
                move_recommendation = MoveRecommendation(move=best_move)

            else:
                print('Playing with player (not Syzygy)')
                move_recommendation = self.main_move_selector.select_move(board)

        return move_recommendation
