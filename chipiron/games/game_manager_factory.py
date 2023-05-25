import chess
import queue

import chipiron as ch
from chipiron.chessenvironment.board.factory import create_board
import chipiron.players as players
from chipiron.players.factory import launch_player_process
from .game_manager import GameManager
from .game import Game, ObservableGame


class GameManagerFactory:
    """
    The GameManagerFactory creates GameManager once the players and rules have been decided.
    Calling create ask for the creation of a GameManager depending on args and players.
    This class is supposed to be independent of Match-related classes (contrarily to the GameArgsFactory)
    """

    def __init__(self,
                 syzygy_table: object,
                 game_manager_board_evaluator_factory: object,
                 output_folder_path: str,
                 main_thread_mailbox: queue.Queue) -> None:
        self.syzygy_table = syzygy_table
        self.output_folder_path = output_folder_path
        self.game_manager_board_evaluator_factory = game_manager_board_evaluator_factory
        self.main_thread_mailbox = main_thread_mailbox
        self.subscribers = []

    def create(
            self,
            args_game_manager: dict,
            player_color_to_player: dict
    ) -> GameManager:
        # maybe this factory is overkill at the moment but might be
        # useful if the logic of game generation gets more complex

        board: ch.chess.BoardChi = create_board()
        player_color_to_id: dict = {color: player.id for color, player in player_color_to_player.items()}
        if self.subscribers:
            for subscriber in self.subscribers:
                player_id_message: dict = {'type': 'players_color_to_id',
                                           'players_color_to_id': player_color_to_id}
                subscriber.put(player_id_message)
        board_evaluator = self.game_manager_board_evaluator_factory.create()

        while not self.main_thread_mailbox.empty():
            self.main_thread_mailbox.get()

        # creating the game playing status
        game_playing_status: ch.games.GamePlayingStatus = ch.games.GamePlayingStatus()

        game: Game = ch.games.Game(playing_status=game_playing_status, board=board)
        obs_game: ObservableGame = ch.games.ObservableGame(game)

        if self.subscribers:
            for subscriber in self.subscribers:
                print('lplpl')
                obs_game.register_mailbox(subscriber, 'board_to_display')

        player_processes = []
        # Creating and launching the player threads
        for player_color in chess.COLORS:
            player = player_color_to_player[player_color]
            game_player = players.GamePlayer(player, player_color)
            if player.id != 'Human':  # TODO COULD WE DO BETTER ? maybe with the null object
                player_process = launch_player_process(game_player,
                                                       obs_game,
                                                       self.main_thread_mailbox)
                player_processes.append(player_process)

        game_manager: GameManager
        game_manager = GameManager(game=obs_game,
                                   syzygy=self.syzygy_table,
                                   display_board_evaluator=board_evaluator,
                                   output_folder_path=self.output_folder_path,
                                   args=args_game_manager,
                                   player_color_to_id=player_color_to_id,
                                   main_thread_mailbox=self.main_thread_mailbox,
                                   player_threads=player_processes)

        return game_manager

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
        self.game_manager_board_evaluator_factory.subscribers.append(subscriber)
