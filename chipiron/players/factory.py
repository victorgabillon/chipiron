import players.move_selector.treevalue as treevalue
from .player import Player
from .player_thread import PlayerProcess
from chipiron.utils.null_object import NullObject

import multiprocessing
import random

from dataclasses import dataclass

import move_selector


@dataclass
class PlayerArgs:
    name: str
    main_move_selector: move_selector.AllMoveSelectorArgs
    # whether to play with syzygy when possible
    syzygy_play: bool


def create_main_move_selector(
        arg: treevalue.TreeAndValuePlayerArgs,
        syzygy,
        random_generator: random.Random
) -> move_selector.MoveSelector:
    main_move_selector: move_selector.MoveSelector
    print('create main move')
    match arg:
        case 'RandomPlayer':
            main_move_selector = move_selector.Random()
        case treevalue.TreeAndValuePlayerArgs():
            main_move_selector = treevalue.create_tree_and_value_builders(args=arg,
                                                                          syzygy=syzygy,
                                                                          random_generator=random_generator)
        case 'Stockfish':
            main_move_selector = move_selector.Stockfish(arg)
        case 'Human':
            main_move_selector = NullObject()  # TODO is it necessary?
        case other:
            raise Exception(f'player creator: can not find {other} {type(other)}')
    return main_move_selector


def create_player(args: PlayerArgs,
                  syzygy,
                  random_generator: random.Random) -> Player:
    print('create player')
    main_move_selector: move_selector.MoveSelector = create_main_move_selector(arg=args.main_move_selector,
                                                                               syzygy=syzygy,
                                                                               random_generator=random_generator)
    return Player(name=args.name,
                  syzygy_play=args.syzygy_play,  # looks like double arguments change?
                  syzygy=syzygy,
                  main_move_selector=main_move_selector)


def create_player_thread(args: dict,
                         syzygy,
                         random_generator: random.Random):
    player = create_player(args, syzygy, random_generator)
    return PlayerProcess(player)


def launch_player_process(game_player,
                          observable_game,
                          main_thread_mailbox):
    # creating objects Queue that is the mailbox for the player thread
    player_thread_mailbox = multiprocessing.Manager().Queue()

    # registering to the observable board to get notification when it changes
    observable_game.register_mailbox(player_thread_mailbox, 'board_to_play')

    # creating and starting the thread for the player
    player_thread = PlayerProcess(game_player, player_thread_mailbox, main_thread_mailbox)
    player_thread.start()

    return player_thread
