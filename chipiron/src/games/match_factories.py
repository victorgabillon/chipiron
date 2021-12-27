from src.players.player import Player
from src.chessenvironment.boards.board import BoardChi
from src.games.game_manager import GameManager
from src.games.match_manager import MatchManager



def create_match_manager(args_match, args_player_one, args_player_two, syzygy_table, output_folder_path):
    player_one = Player(args_player_one, syzygy_table)
    player_two = Player(args_player_two, syzygy_table)
    board = BoardChi()
    game_manager = GameManager(board, syzygy_table, output_folder_path)
    match_manager = MatchManager(args_match, player_one, player_two, game_manager, output_folder_path)

    return match_manager


