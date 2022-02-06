import sys
from src.players.random_player import RandomPlayer
from src.players.treevaluebuilders.create_tree_and_value import create_tree_and_value_builders
from src.players.stockfish import Stockfish
from src.players.player import Player
from src.players.player_thread import PlayerProcess
import multiprocessing
from src.extra_tools.null_object import NullObject


def create_main_move_selector(arg, syzygy, random_generator):
    if arg['type'] == 'RandomPlayer':
        main_move_selector = RandomPlayer()
    elif arg['type'] == 'TreeAndValue':
        main_move_selector = create_tree_and_value_builders(arg, syzygy, random_generator)
    elif arg['type'] == 'Stockfish':
        main_move_selector = Stockfish(arg)
    else:
        sys.exit('player creator: can not find ' + arg['type'])
    return main_move_selector


def create_player(args, syzygy, random_generator):
    main_move_selector = create_main_move_selector(args, syzygy, random_generator)
    return Player(args, syzygy, main_move_selector)


def create_player_thread(args, syzygy, random_generator):
    player = create_player(args, syzygy, random_generator)
    return PlayerProcess(player)


def launch_player_process(game_player, observable_board, main_thread_mailbox):
    # creating objects Queue that is the mailbox for the player thread
    player_thread_mailbox = multiprocessing.Manager().Queue()

    # registering to the observable board to get notification when it changes
    observable_board.register_mailbox(player_thread_mailbox)

    # creating and starting the thread for the player
    player_thread = PlayerProcess(game_player, player_thread_mailbox, main_thread_mailbox)
    player_thread.start()

    return player_thread
