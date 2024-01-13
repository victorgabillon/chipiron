from .player import Player
from .player_thread import PlayerProcess
from .game_player import GamePlayer, game_player_computes_move_on_board_and_send_move_in_queue

import multiprocessing
import random
import queue

from dataclasses import dataclass
from functools import partial

from . import move_selector

from chipiron.environments.chess import BoardChi
from typing import Callable

from chipiron.utils.communication.player_game_messages import BoardMessage


@dataclass
class PlayerArgs:
    name: str
    main_move_selector: move_selector.AllMoveSelectorArgs
    # whether to play with syzygy when possible
    syzygy_play: bool


def create_player(args: PlayerArgs,
                  syzygy,
                  random_generator: random.Random) -> Player:
    print('create player')
    main_move_selector: move_selector.MoveSelector = move_selector.create_main_move_selector(
        move_selector_instance_or_args=args.main_move_selector,
        syzygy=syzygy,
        random_generator=random_generator
    )
    return Player(name=args.name,
                  syzygy_play=args.syzygy_play,  # looks like double arguments change?
                  syzygy=syzygy,
                  main_move_selector=main_move_selector)


def send_board_to_player_process_mailbox(
        board: BoardChi,
        player_process_mailbox: queue.Queue
) -> None:
    message: BoardMessage = BoardMessage(board=board)
    player_process_mailbox.put(item=message)


MoveFunction = Callable[[BoardChi], None]


def create_player_observer(
        game_player: GamePlayer,
        distributed_players: bool,
        main_thread_mailbox: queue.Queue
) -> tuple[GamePlayer | PlayerProcess, MoveFunction]:
    generic_player: GamePlayer | PlayerProcess
    move_function: MoveFunction
    if distributed_players:
        # creating objects Queue that is the mailbox for the player thread
        player_process_mailbox = multiprocessing.Manager().Queue()

        # creating and starting the thread for the player
        player_process: PlayerProcess = PlayerProcess(game_player=game_player,
                                                      queue_board=player_process_mailbox,
                                                      queue_move=main_thread_mailbox)
        player_process.start()
        generic_player = player_process

        move_function = partial(send_board_to_player_process_mailbox, player_process_mailbox=player_process_mailbox)

    else:
        generic_player = game_player
        move_function = partial(
            game_player_computes_move_on_board_and_send_move_in_queue,
            game_player=game_player,
            queue_move=main_thread_mailbox
        )

    return generic_player, move_function
