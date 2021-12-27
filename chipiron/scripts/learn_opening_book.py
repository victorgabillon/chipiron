import pickle
import yaml
from src.opening_book import OpeningBook
from src.games.match_manager import MatchManager
from src.players import factory
from src.chessenvironment.chess_environment import ChessEnvironment
from src.players.boardevaluators.syzygy import SyzygyTable
import global_variables

try:
    with open('src/opening_book/opening_book.data', 'rb') as file_opening_book:
        opening_book = pickle.load(file_opening_book)
except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
    opening_book = OpeningBook()

file_name_player_one = 'RecurZipf2.yaml'
file_name_match_setting = 'setting_jime.yaml'
path_player_one = 'runs/players/' + file_name_player_one
path_match_setting = 'runs/OneMatch/' + file_name_match_setting

with open(path_match_setting, 'r') as fileMatch:
    args_match = yaml.load(fileMatch, Loader=yaml.FullLoader)
    print(args_match)

with open(path_player_one, 'r') as filePlayerOne:
    args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
    print(args_player_one)

fileGame = args_match['gameSettingFile']
path_game_setting = 'runs/GameSettings/' + fileGame

chess_simulator = ChessEnvironment()
syzygy = SyzygyTable(chess_simulator)

player_one = create_player(args_player_one, chess_simulator, syzygy)
player_two = create_player(args_player_one, chess_simulator, syzygy)
global_variables.init()  # global variables

while True:
    starting_position = opening_book.get_opening_position_to_learn()
    play = MatchManager(args_match, player_one, player_two, chess_simulator, syzygy)
    p1wins, p2wins, draws = play.play_the_match()
    opening_book.position_result(starting_position, p1wins, p2wins, draws)
    with open('src/opening_book/opening_book.data', 'rb') as file_opening_book:
        pickle.load(opening_book,file_opening_book)
