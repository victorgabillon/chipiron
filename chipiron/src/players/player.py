class Player():
    #  difference between player and treebuilder includes the fact that now a player can be a mixture of multiple decision rulles

    def __init__(self):
        pass

    def get_move(self, board, time):
        """ returns the best move computed by the player.
        The player has the option to ask the syzygy table to play it"""

        # if there is only one possible legal move in the position, do not think, choose it.
        all_legal_moves = list(board.get_legal_moves())
        if len(all_legal_moves) == 1 and self.player_name != 'Human':
            return all_legal_moves[0]

        # if the play with syzygy option is on test if the position is in the database to play syzygy
        if self.syzygy_play and self.syzygy_player.fast_in_table(board):
            print('Playing with Syzygy')
            best_move = self.syzygy_player.best_move(board)

        else:
            print('Playing with player (not Syzygy)')
            best_move = self.get_move_from_player(board, time)

        return best_move


    def print_info(self):
        pass
        # print('------------\nPlayer ',self.color)
