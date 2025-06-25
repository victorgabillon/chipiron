"""
Module for the GameManagerFactory class.
"""

import queue
from dataclasses import dataclass, field
from typing import Any

import chess

import chipiron as ch
import chipiron.environments.chess.board as boards
import chipiron.players as players_m
from chipiron.games.game.game_args import GameArgs
from chipiron.players import PlayerFactoryArgs
from chipiron.players.boardevaluators.board_evaluator import (
    IGameBoardEvaluator,
    ObservableBoardEvaluator,
)
from chipiron.players.factory_higher_level import (
    MoveFunction,
    PlayerObserverFactory,
    create_player_observer_factory,
)
from chipiron.utils import path, seed
from chipiron.utils.communication.gui_player_message import PlayersColorToPlayerMessage
from chipiron.utils.dataclass import IsDataclass

from ...environments.chess.board.utils import FenPlusHistory
from ...environments.chess.move_factory import MoveFactory
from ...players.boardevaluators.table_base import SyzygyTable
from ...players.player_ids import PlayerConfigTag
from ...scripts.chipiron_args import ImplementationArgs
from .game import Game, ObservableGame
from .game_manager import GameManager
from .progress_collector import PlayerProgressCollectorObservable


@dataclass
class GameManagerFactory:
    """
    The GameManagerFactory creates GameManager once the players and rules have been decided.
    Calling create ask for the creation of a GameManager depending on args and players.
    This class is supposed to be independent of Match-related classes (contrarily to the GameArgsFactory)

    Args:
    syzygy_table (SyzygyTable | None): The syzygy table used for endgame tablebase lookups.
    game_manager_board_evaluator (IGameBoardEvaluator): The game board evaluator used for evaluating game positions.
    output_folder_path (path | None): The path to the output folder where game data will be saved.
    main_thread_mailbox (queue.Queue[IsDataclass]): The mailbox used for communication between processes.

    """

    # todo we might want to plit this into various part, like maybe a player factory, not sure, think about it

    syzygy_table: SyzygyTable[Any] | None
    output_folder_path: path | None
    main_thread_mailbox: queue.Queue[IsDataclass]
    game_manager_board_evaluator: IGameBoardEvaluator
    board_factory: boards.BoardFactory
    move_factory: MoveFactory
    implementation_args: ImplementationArgs
    universal_behavior: bool
    subscribers: list[queue.Queue[IsDataclass]] = field(default_factory=list)

    def create(
        self,
        args_game_manager: GameArgs,
        player_color_to_factory_args: dict[chess.Color, PlayerFactoryArgs],
        game_seed: seed,
    ) -> GameManager:
        """
        Create a GameManager with the given arguments

        Args:
            args_game_manager (GameArgs): the arguments of the game manager
            player_color_to_factory_args (dict[chess.Color, PlayerFactoryArgs]): the arguments of the players
            game_seed (int): the seed of the game

        Returns:
            the created GameManager
        """
        # useful if the logic of game generation gets complex
        # in the future, we might want the implementation detail to actually be modified during the
        # match in that case they would come arg_game_manager

        # CREATING THE BOARD
        starting_fen: str = args_game_manager.starting_position.get_fen()
        board: boards.IBoard = self.board_factory(
            fen_with_history=FenPlusHistory(current_fen=starting_fen)
        )
        if self.subscribers:
            for subscriber in self.subscribers:
                player_id_message: PlayersColorToPlayerMessage = (
                    PlayersColorToPlayerMessage(
                        player_color_to_factory_args=player_color_to_factory_args
                    )
                )

                subscriber.put(player_id_message)

        while not self.main_thread_mailbox.empty():
            self.main_thread_mailbox.get()

        # creating the game playing status
        game_playing_status: ch.games.GamePlayingStatus = ch.games.GamePlayingStatus()

        game: Game = Game(
            playing_status=game_playing_status, board=board, seed_=game_seed
        )
        observable_game: ObservableGame = ObservableGame(game=game)

        if self.subscribers:
            for subscriber in self.subscribers:
                observable_game.register_display(subscriber)

        # CREATING THE PLAYERS
        player_observer_factory: PlayerObserverFactory = create_player_observer_factory(
            each_player_has_its_own_thread=args_game_manager.each_player_has_its_own_thread,
            implementation_args=self.implementation_args,
            syzygy_table=self.syzygy_table,
            universal_behavior=self.universal_behavior,
        )

        player_progress_collector: PlayerProgressCollectorObservable = (
            PlayerProgressCollectorObservable(subscribers=self.subscribers)
        )

        players: list[players_m.GamePlayer | players_m.PlayerProcess] = []
        # Creating the players
        for player_color in chess.COLORS:
            player_factory_args: players_m.PlayerFactoryArgs = (
                player_color_to_factory_args[player_color]
            )

            # Human playing with gui does not need a player, as the playing moves will be generated directly
            # by the GUI and sent directly to the game_manager
            if player_factory_args.player_args.name != PlayerConfigTag.GUI_HUMAN:
                generic_player: players_m.GamePlayer | players_m.PlayerProcess
                move_function: MoveFunction
                generic_player, move_function = player_observer_factory(
                    player_color=player_color,
                    player_factory_args=player_factory_args,
                    main_thread_mailbox=self.main_thread_mailbox,
                    # player_progress_collector=player_progress_collector
                )
                players.append(generic_player)

                # registering to the observable board to get notification when it changes
                observable_game.register_player(move_function=move_function)

        player_color_to_id: dict[chess.Color, str] = {
            color: player_factory_args.player_args.name
            for color, player_factory_args in player_color_to_factory_args.items()
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
            move_factory=self.move_factory,
            progress_collector=player_progress_collector,
        )

        return game_manager

    def subscribe(self, subscriber: queue.Queue[IsDataclass]) -> None:
        """
        Subscribe to the GameManagerFactory to get the PlayersColorToPlayerMessage
        As well as subscribing to the game_manager_board_evaluator to get the EvaluationMessage

        Args:
            subscriber: the subscriber queue
        """
        self.subscribers.append(subscriber)
        assert isinstance(self.game_manager_board_evaluator, ObservableBoardEvaluator)
        self.game_manager_board_evaluator.subscribe(subscriber)
