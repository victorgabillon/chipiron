import global_variables
from src.players.RandomPlayer import RandomPlayer
from src.players.treevaluebuilders.create_tree_and_value import create_tree_and_value_builders
from src.players.human import Human
from src.players.stockfish import Stockfish
import sys


class Player:
    #  difference between player and treebuilder includes the fact that now a player can be a mixture of multiple decision rulles

    def __init__(self, arg, syzygy):
        print(arg)
        self.arg = arg
        self.player_name_id = arg['name']

        self.main_move_selector = None
        if arg['type'] == 'RandomPlayer':
            self.main_move_selector = RandomPlayer()
        elif arg['type'] == 'TreeAndValue':
            self.main_move_selector = create_tree_and_value_builders(arg, syzygy)
        elif arg['type'] == 'Human':
            self.main_move_selector = Human(arg)
        elif arg['type'] == 'Stockfish':
            self.main_move_selector = Stockfish(arg)
        else:
            sys.exit('player creator: can not find ' + arg['type'])

        self.syzygy_player = None
        self.syzygy_play = arg['syzygy_play']
        if self.syzygy_play:
            self.syzygy_player = syzygy

        self.print_info()

    def get_move(self, board, time):
        """ returns the best move computed by the player.
        The player has the option to ask the syzygy table to play it"""

        # if there is only one possible legal move in the position, do not think, choose it.
        all_legal_moves = list(board.legal_moves)
        if len(all_legal_moves) == 1 and self.player_name_id != 'Human':
            return all_legal_moves[0]

        # if the play with syzygy option is on test if the position is in the database to play syzygy
        if self.syzygy_play and self.syzygy_player.fast_in_table(board):
            print('Playing with Syzygy')
            best_move = self.syzygy_player.best_move(board)

        else:
            print('Playing with player (not Syzygy)')
            best_move = self.main_move_selector.get_move_from_player(board, time)

        return best_move

    def print_info(self):
        pass
        # print('------------\nPlayer ',self.color)
