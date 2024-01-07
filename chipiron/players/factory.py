from .player import Player
from .player_thread import PlayerProcess
from .game_player import GamePlayer

import multiprocessing
import random

from dataclasses import dataclass

from . import move_selector


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
        args=args.main_move_selector,
        syzygy=syzygy,
        random_generator=random_generator
    )
    return Player(name=args.name,
                  syzygy_play=args.syzygy_play,  # looks like double arguments change?
                  syzygy=syzygy,
                  main_move_selector=main_move_selector)


def create_player_thread(args: dict,
                         syzygy,
                         random_generator: random.Random):
    player = create_player(args, syzygy, random_generator)
    return PlayerProcess(player)


def launch_player_process(
        game_player: GamePlayer,
        observable_game,
        main_thread_mailbox
) -> PlayerProcess:
    # creating objects Queue that is the mailbox for the player thread
    player_thread_mailbox = multiprocessing.Manager().Queue()

    # registering to the observable board to get notification when it changes
    observable_game.register_mailbox(player_thread_mailbox, 'board_to_play')

    # creating and starting the thread for the player
    player_thread: PlayerProcess = PlayerProcess(game_player=game_player,
                                                 queue_board=player_thread_mailbox,
                                                 queue_move=main_thread_mailbox)
    player_thread.start()

    return player_thread
