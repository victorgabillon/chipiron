from scripts.script import Script
from src.players.create_player import create_player
import yaml
from src.chessenvironment.chess_environment import ChessEnvironment
from src.players.boardevaluators.syzygy import Syzygy

class LearnNNTreeStrap(Script):

    def __init__(self):
        super().__init__()
        file_name_player_one = 'RecurZipfBase.yaml'
        path_player_one = 'chipiron/runs/players/' + file_name_player_one

        with open(path_player_one, 'r') as filePlayerOne:
            args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
            print(args_player_one)

        chess_simulator = ChessEnvironment()
        syzygy = Syzygy(chess_simulator, '')
        self.player_one = create_player(args_player_one, chess_simulator, syzygy)

    def run(self):
        print('run')
        for
        tree_strap_one_board(board, tree_value_builder):
