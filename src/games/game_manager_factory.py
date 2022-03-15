from src.chessenvironment.board.factory import create_board
from src.games.game_manager import GameManager
import src.players as players
import chess
from src.players.factory import launch_player_process


class GameManagerFactory:
    """
    The GameManagerFactory creates GameManager once the players and rules have been decided.
    Calling create ask for the creation of a GameManager depending on args and players.
    This class is supposed to be independent of Match-related classes (contrarily to the GameArgsFactory)
    """

    def __init__(self,
                 syzygy_table: object,
                 game_manager_board_evaluator_factory: object,
                 output_folder_path: object,
                 main_thread_mailbox: object) -> None:
        self.syzygy_table = syzygy_table
        self.output_folder_path = output_folder_path
        self.game_manager_board_evaluator_factory = game_manager_board_evaluator_factory
        self.main_thread_mailbox = main_thread_mailbox
        self.subscribers = []

    def create(self, args_game_manager, player_color_to_player):
        # maybe this factory is overkill at the moment but might be
        # useful if the logic of game generation gets more complex

        board = create_board(self.subscribers)
        player_color_to_id = {color: player.id for color, player in player_color_to_player.items()}
        if self.subscribers:
            for subscriber in self.subscribers:
                subscriber.put({'type': 'players_color_to_id', 'players_color_to_id': player_color_to_id})
        board_evaluator = self.game_manager_board_evaluator_factory.create()

        while not self.main_thread_mailbox.empty():
            self.main_thread_mailbox.get()

        player_processes = []
        # Creating and launching the player threads
        for player_color in chess.COLORS:
            player = player_color_to_player[player_color]
            game_player = players.GamePlayer(player, player_color)
            if player.id != 'Human':  # TODO COULD WE DO BETTER ? maybe with the null object
                player_process = launch_player_process(game_player, board, self.main_thread_mailbox)
                player_processes.append(player_process)

        game_manager = GameManager(board,
                                   self.syzygy_table,
                                   board_evaluator,
                                   self.output_folder_path,
                                   args_game_manager,
                                   player_color_to_id,
                                   self.main_thread_mailbox,
                                   player_processes)

        return game_manager

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
        self.game_manager_board_evaluator_factory.subscribers.append(subscriber)
