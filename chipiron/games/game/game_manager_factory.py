import queue

import chess

import chipiron as ch
import chipiron.environments.chess.board as boards
import chipiron.players as players_m
from chipiron.games.game.game_args import GameArgs
from chipiron.players import PlayerFactoryArgs
from chipiron.players.boardevaluators.board_evaluator import IGameBoardEvaluator, ObservableBoardEvaluator
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.factory_higher_level import create_player_observer
from chipiron.utils import path
from chipiron.utils import seed
from chipiron.utils.communication.gui_player_message import PlayersColorToPlayerMessage, extract_message_from_players
from chipiron.utils.is_dataclass import IsDataclass
from .game import Game, ObservableGame, MoveFunction
from .game_manager import GameManager


class GameManagerFactory:
    """
    The GameManagerFactory creates GameManager once the players and rules have been decided.
    Calling create ask for the creation of a GameManager depending on args and players.
    This class is supposed to be independent of Match-related classes (contrarily to the GameArgsFactory)
    """
    syzygy_table: SyzygyTable | None
    subscribers: list[queue.Queue[IsDataclass]]

    def __init__(
            self,
            syzygy_table: SyzygyTable | None,
            game_manager_board_evaluator: IGameBoardEvaluator,
            output_folder_path: path | None,
            main_thread_mailbox: queue.Queue[IsDataclass],
    ) -> None:
        self.syzygy_table = syzygy_table
        self.output_folder_path = output_folder_path
        self.game_manager_board_evaluator = game_manager_board_evaluator
        self.main_thread_mailbox = main_thread_mailbox
        self.subscribers = []

    def create(
            self,
            args_game_manager: GameArgs,
            player_color_to_factory_args: dict[chess.Color, PlayerFactoryArgs],
            game_seed: seed
    ) -> GameManager:
        # maybe this factory is overkill at the moment but might be
        # useful if the logic of game generation gets more complex

        board: boards.BoardChi = boards.create_board()
        if self.subscribers:
            for subscriber in self.subscribers:
                player_id_message: PlayersColorToPlayerMessage = extract_message_from_players(
                    player_color_to_factory_args=player_color_to_factory_args
                )
                subscriber.put(player_id_message)

        while not self.main_thread_mailbox.empty():
            self.main_thread_mailbox.get()

        # creating the game playing status
        game_playing_status: ch.games.GamePlayingStatus = ch.games.GamePlayingStatus()

        game: Game = Game(
            playing_status=game_playing_status,
            board=board,
            seed_=game_seed
        )
        observable_game: ObservableGame = ObservableGame(game=game)

        if self.subscribers:
            for subscriber in self.subscribers:
                observable_game.register_display(subscriber)

        players: list[players_m.GamePlayer | players_m.PlayerProcess] = []
        # Creating the players
        for player_color in chess.COLORS:
            player_factory_args: players_m.PlayerFactoryArgs = player_color_to_factory_args[player_color]

            # Human playing with gui does not need a player, as the playing moves will be generated directly
            # by the GUI and sent directly to the game_manager
            if player_factory_args.player_args.name != 'Gui_Human':
                generic_player: players_m.GamePlayer | players_m.PlayerProcess
                move_function: MoveFunction
                generic_player, move_function = create_player_observer(
                    player_color=player_color,
                    player_factory_args=player_factory_args,
                    distributed_players=args_game_manager.each_player_has_its_own_thread,
                    main_thread_mailbox=self.main_thread_mailbox
                )
                players.append(generic_player)

                # registering to the observable board to get notification when it changes
                observable_game.register_player(move_function=move_function)

        player_color_to_id: dict[chess.Color, str] = {
            color: player_factory_args.player_args.name for color, player_factory_args in
            player_color_to_factory_args.items()
        }

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
        )

        return game_manager

    def subscribe(
            self,
            subscriber: queue.Queue[IsDataclass]
    ) -> None:
        self.subscribers.append(subscriber)
        assert isinstance(self.game_manager_board_evaluator, ObservableBoardEvaluator)
        self.game_manager_board_evaluator.subscribe(subscriber)
