import chipiron.players as players
import chipiron.players.treevalue as treevalue

from chipiron.extra_tools.null_object import NullObject

import multiprocessing
import random


def create_main_move_selector(arg: dict,
                              syzygy,
                              random_generator: random.Random) -> players.MoveSelector:
    main_move_selector: players.MoveSelector
    match arg['type']:
        case 'RandomPlayer':
            main_move_selector = players.RandomPlayer()
        case 'TreeAndValue':
            main_move_selector = players.treevalue.create_tree_and_value_builders(arg=arg,
                                                                                  syzygy=syzygy,
                                                                                  random_generator=random_generator)
        case 'Stockfish':
            main_move_selector = players.Stockfish(arg)
        case 'Human':
            main_move_selector = NullObject()  # TODO is it necessary?
        case other:
            raise Exception(f'player creator: can not find {other}')
    return main_move_selector


def create_player(args: dict,
                  syzygy,
                  random_generator: random.Random) -> players.Player:
    main_move_selector: players.MoveSelector = create_main_move_selector(arg=args, syzygy=syzygy,
                                                                         random_generator=random_generator)
    return players.Player(arg=args,
                          syzygy=syzygy,
                          main_move_selector=main_move_selector)


def create_player_thread(args: dict,
                         syzygy,
                         random_generator: random.Random):
    player = create_player(args, syzygy, random_generator)
    return players.PlayerProcess(player)


def launch_player_process(game_player,
                          observable_game,
                          main_thread_mailbox):
    # creating objects Queue that is the mailbox for the player thread
    player_thread_mailbox = multiprocessing.Manager().Queue()

    # registering to the observable board to get notification when it changes
    observable_game.register_mailbox(player_thread_mailbox, 'board_to_play')

    # creating and starting the thread for the player
    player_thread = players.PlayerProcess(game_player, player_thread_mailbox, main_thread_mailbox)
    player_thread.start()

    return player_thread
