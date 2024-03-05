from .player import Player
from .player_thread import PlayerProcess
from .game_player import GamePlayer, game_player_computes_move_on_board_and_send_move_in_queue

import multiprocessing
import random
import queue

from functools import partial

from . import move_selector

from chipiron.environments.chess import BoardChi
from typing import Callable

from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils import seed
from chipiron.players.boardevaluators.table_base.factory import create_syzygy, SyzygyTable

from chipiron.players.utils import fetch_player_args_convert_and_save
from .player_args import PlayerArgs


def create_chipiron_player(
        depth: int
) -> Player:
    syzygy_table: SyzygyTable | None = create_syzygy()
    random_generator = random.Random()

    args_player: PlayerArgs = fetch_player_args_convert_and_save(
        file_name_player='data/players/player_config/chipiron/chipiron.yaml',
        from_data_folder=False)

    main_move_selector: move_selector.MoveSelector = move_selector.create_main_move_selector(
        move_selector_instance_or_args=args_player.main_move_selector,
        syzygy=syzygy_table,
        random_generator=random_generator
    )

    return Player(
        name='chipiron',
        syzygy=syzygy_table,
        main_move_selector=main_move_selector
    )


def create_player(
        args: PlayerArgs,
        syzygy,
        random_generator: random.Random
) -> Player:
    print('create player')
    main_move_selector: move_selector.MoveSelector = move_selector.create_main_move_selector(
        move_selector_instance_or_args=args.main_move_selector,
        syzygy=syzygy,
        random_generator=random_generator
    )
    return Player(
        name=args.name,
        syzygy=syzygy,
        main_move_selector=main_move_selector
    )


def send_board_to_player_process_mailbox(
        board: BoardChi,
        seed_: int,
        player_process_mailbox: queue.Queue[BoardMessage]
) -> None:
    message: BoardMessage = BoardMessage(
        board=board,
        seed=seed_
    )
    player_process_mailbox.put(item=message)


MoveFunction = Callable[[BoardChi, seed], None]


def create_player_observer(
        game_player: GamePlayer,
        distributed_players: bool,
        main_thread_mailbox: queue.Queue
) -> tuple[GamePlayer | PlayerProcess, MoveFunction]:
    generic_player: GamePlayer | PlayerProcess
    move_function: MoveFunction

    # case with multiprocessing when each player is a separate process
    if distributed_players:
        # creating objects Queue that is the mailbox for the player thread
        player_process_mailbox = multiprocessing.Manager().Queue()

        # creating and starting the thread for the player
        player_process: PlayerProcess = PlayerProcess(
            game_player=game_player,
            queue_board=player_process_mailbox,
            queue_move=main_thread_mailbox
        )
        player_process.start()
        generic_player = player_process

        move_function = partial(
            send_board_to_player_process_mailbox,
            player_process_mailbox=player_process_mailbox

        )

    # case without multiprocessing all players and match manager in the same process
    else:
        generic_player = game_player
        move_function = partial(
            game_player_computes_move_on_board_and_send_move_in_queue,
            game_player=game_player,
            queue_move=main_thread_mailbox
        )

    return generic_player, move_function
