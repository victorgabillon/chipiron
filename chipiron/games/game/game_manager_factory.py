"""
Module for the GameManagerFactory class.
"""

import queue
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import atomheart.board as boards
from atomheart.move_factory import MoveFactory
from valanga import Color, TurnState
from valanga.game import Seed

from chipiron.displays.gui_protocol import (
    GuiUpdate,
    MatchId,
    Scope,
    SessionId,
    make_scope,
    scope_for_new_game,
)
from chipiron.games.game.game_args import GameArgs
from chipiron.games.game.game_playing_status import GamePlayingStatus
from chipiron.players import PlayerFactoryArgs
from chipiron.players.boardevaluators.board_evaluator import (
    IGameStateEvaluator,
    ObservableGameStateEvaluator,
)
from chipiron.environments.environment import EnvironmentDeps, make_environment
from chipiron.players.factory_higher_level import MoveFunction, PlayerObserverFactory
from chipiron.utils import path
from chipiron.utils.communication.gui_messages.gui_messages import (
    make_players_info_payload,
)
from chipiron.displays.gui_publisher import GuiPublisher

from ...players.boardevaluators.table_base.factory import AnySyzygyTable
from ...players.player_ids import PlayerConfigTag
from ...scripts.chipiron_args import ImplementationArgs
from .game import Game, ObservableGame
from .game_manager import GameManager, MainMailboxMessage
from .progress_collector import PlayerProgressCollectorObservable

if TYPE_CHECKING:
    import chipiron.players as players_m


def make_subscriber_queues() -> list[queue.Queue[GuiUpdate]]:
    return []


@dataclass
class GameManagerFactory[StateT: TurnState]:
    """
    The GameManagerFactory creates GameManager once the players and rules have been decided.
    Calling create ask for the creation of a GameManager depending on args and players.
    This class is supposed to be independent of Match-related classes (contrarily to the GameArgsFactory)

    Args:
    syzygy_table (AnySyzygyTable | None): The syzygy table used for endgame tablebase lookups.
    game_manager_board_evaluator (IGameBoardEvaluator): The game board evaluator used for evaluating game positions.
    output_folder_path (path | None): The path to the output folder where game data will be saved.
    main_thread_mailbox (queue.Queue[MainMailboxMessage]): The mailbox used for communication between processes.

    """

    # todo we might want to plit this into various part, like maybe a player factory, not sure, think about it

    syzygy_table: AnySyzygyTable | None
    output_folder_path: path | None
    main_thread_mailbox: queue.Queue[MainMailboxMessage]
    game_manager_state_evaluator: IGameStateEvaluator[StateT]
    board_factory: boards.BoardFactory
    move_factory: MoveFactory
    implementation_args: ImplementationArgs
    universal_behavior: bool
    subscriber_queues: list[queue.Queue[GuiUpdate]] = make_subscriber_queues()
    session_id: SessionId = ""
    match_id: MatchId | None = None

    def create(
        self,
        args_game_manager: GameArgs,
        player_color_to_factory_args: dict[Color, PlayerFactoryArgs],
        game_seed: Seed,
    ) -> GameManager[StateT]:
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

        if not self.session_id:
            raise ValueError("GameManagerFactory.session_id must be set")

        game_id: str = uuid.uuid4().hex

        base_scope = make_scope(
            session_id=self.session_id,
            match_id=self.match_id,
            game_id="",
        )

        scope: Scope = scope_for_new_game(base_scope, game_id)

        publishers: list[GuiPublisher] = [
            GuiPublisher(
                out=q,
                schema_version=1,
                game_kind=args_game_manager.game_kind,
                scope=scope,
            )
            for q in self.subscriber_queues
        ]

        # If the evaluator is observable, avoid mutating the shared instance.
        # Create a per-game wrapper so publisher scoping is isolated.
        display_state_evaluator: IGameStateEvaluator[StateT]
        if isinstance(self.game_manager_state_evaluator, ObservableGameStateEvaluator):
            display_state_evaluator = ObservableGameStateEvaluator(
                game_state_evaluator=self.game_manager_state_evaluator.game_state_evaluator
            )
            for pub in publishers:
                display_state_evaluator.subscribe(pub)
        else:
            display_state_evaluator = self.game_manager_state_evaluator

        environment = make_environment(
            game_kind=args_game_manager.game_kind,
            syzygy_table=self.syzygy_table,
            deps=EnvironmentDeps(board_factory=self.board_factory),
        )

        start_tag = args_game_manager.starting_position.get_start_tag()
        normalized_start_tag = environment.normalize_start_tag(start_tag)
        board = environment.make_initial_state(normalized_start_tag)
        for publisher in publishers:
            payload = make_players_info_payload(
                player_color_to_factory_args=player_color_to_factory_args
            )
            publisher.publish(payload)

        # Do not drain the shared mailbox here.
        # GameManager already ignores stale messages via scope filtering.

        # creating the game playing status
        game_playing_status: GamePlayingStatus = GamePlayingStatus()

        game: Game[StateT] = Game[StateT](
            playing_status=game_playing_status, state=board, seed_=game_seed
        )

        observable_game: ObservableGame[StateT] = ObservableGame[StateT](
            game=game,
            gui_encoder=environment.gui_encoder,
            player_encoder=environment.player_encoder,
            scope=scope,
        )

        for pub in publishers:
            observable_game.register_display(pub)

        # CREATING THE PLAYERS
        player_observer_factory: PlayerObserverFactory = environment.make_player_observer_factory(
            each_player_has_its_own_thread=args_game_manager.each_player_has_its_own_thread,
            implementation_args=self.implementation_args,
            universal_behavior=self.universal_behavior,
        )

        player_progress_collector: PlayerProgressCollectorObservable = (
            PlayerProgressCollectorObservable(publishers=publishers)
        )

        players: list[players_m.PlayerHandle] = []
        # Creating the players
        for player_color in (Color.WHITE, Color.BLACK):
            player_factory_args: players_m.PlayerFactoryArgs = (
                player_color_to_factory_args[player_color]
            )

            # Human playing with gui does not need a player, as the playing moves will be generated directly
            # by the GUI and sent directly to the game_manager
            if player_factory_args.player_args.name != PlayerConfigTag.GUI_HUMAN:
                generic_player: players_m.PlayerHandle
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

        player_color_to_id: dict[Color, str] = {
            color: player_factory_args.player_args.name
            for color, player_factory_args in player_color_to_factory_args.items()
        }

        game_manager: GameManager[StateT]
        game_manager = GameManager[StateT](
            game=observable_game,
            display_state_evaluator=display_state_evaluator,
            output_folder_path=self.output_folder_path,
            args=args_game_manager,
            player_color_to_id=player_color_to_id,
            main_thread_mailbox=self.main_thread_mailbox,
            players=players,
            move_factory=self.move_factory,
            progress_collector=player_progress_collector,
            rules=environment.rules,
        )

        return game_manager

    def subscribe(self, subscriber_queue: queue.Queue[GuiUpdate]) -> None:
        """
        Subscribe to the GameManagerFactory to get the PlayersColorToPlayerMessage
        As well as subscribing to the game_manager_board_evaluator to get the EvaluationMessage

        Args:
            subscriber: the subscriber queue
        """
        self.subscriber_queues.append(subscriber_queue)
