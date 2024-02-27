import chess
import queue

import chipiron as ch
from chipiron.environments.chess.board.factory import create_board
import chipiron.players as players_m
from chipiron.players.factory import create_player_observer
from .game_manager import GameManager
from .game import Game, ObservableGame, MoveFunction
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.utils import path
from chipiron.games.game.game_args import GameArgs
from chipiron.utils.communication.gui_player_message import PlayersColorToPlayerMessage, extract_message_from_players
from chipiron.players import Player
from chipiron.utils import seed


class GameManagerFactory:
    """
    The GameManagerFactory creates GameManager once the players and rules have been decided.
    Calling create ask for the creation of a GameManager depending on args and players.
    This class is supposed to be independent of Match-related classes (contrarily to the GameArgsFactory)
    """
    syzygy_table: SyzygyTable

    def __init__(
            self,
            syzygy_table: SyzygyTable,
            game_manager_board_evaluator,
            output_folder_path: path | None,
            main_thread_mailbox: queue.Queue,
            print_svg_board_to_file: bool = False
    ) -> None:
        self.syzygy_table = syzygy_table
        self.output_folder_path = output_folder_path
        self.game_manager_board_evaluator = game_manager_board_evaluator
        self.main_thread_mailbox = main_thread_mailbox
        self.subscribers = []
        self.print_svg_board_to_file=print_svg_board_to_file

    def create(
            self,
            args_game_manager: GameArgs,
            player_color_to_player: dict[chess.COLORS, Player],
            game_seed: seed
    ) -> GameManager:
        # maybe this factory is overkill at the moment but might be
        # useful if the logic of game generation gets more complex

        board: ch.chess.BoardChi = create_board()
        if self.subscribers:
            for subscriber in self.subscribers:
                player_id_message: PlayersColorToPlayerMessage = extract_message_from_players(
                    player_color_to_player=player_color_to_player
                )
                subscriber.put(player_id_message)

        while not self.main_thread_mailbox.empty():
            self.main_thread_mailbox.get()

        # creating the game playing status
        game_playing_status: ch.games.GamePlayingStatus = ch.games.GamePlayingStatus()

        game: Game = Game(
            playing_status=game_playing_status,
            board=board,
            seed=game_seed
        )
        observable_game: ObservableGame = ObservableGame(game=game)

        if self.subscribers:
            for subscriber in self.subscribers:
                observable_game.register_display(subscriber)

        players: list[players_m.PlayerProcess] = []
        # Creating and launching the player threads
        for player_color in chess.COLORS:
            player: players_m.Player = player_color_to_player[player_color]
            game_player: players_m.GamePlayer = players_m.GamePlayer(player, player_color)
            if player.id != 'Human':  # TODO COULD WE DO BETTER ? maybe with the null object
                generic_player: players_m.GamePlayer | players_m.PlayerProcess
                move_function: MoveFunction
                generic_player, move_function = create_player_observer(
                    game_player=game_player,
                    distributed_players=args_game_manager.each_player_has_its_own_thread,
                    main_thread_mailbox=self.main_thread_mailbox
                )
                players.append(generic_player)

                # registering to the observable board to get notification when it changes
                observable_game.register_player(move_function=move_function)

        player_color_to_id: dict = {color: player.id for color, player in player_color_to_player.items()}

        game_manager: GameManager
        game_manager = GameManager(
            game=observable_game,
            syzygy=self.syzygy_table,
            display_board_evaluator=self.game_manager_board_evaluator,
            output_folder_path=self.output_folder_path,
            args=args_game_manager,
            player_color_to_id=player_color_to_id,
            main_thread_mailbox=self.main_thread_mailbox,
            players=players,
            print_svg_board_to_file = self.print_svg_board_to_file
        )

        return game_manager

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
        self.game_manager_board_evaluator.subscribe(subscriber)
