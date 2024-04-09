"""
factory_higher_level.py
"""
import multiprocessing
import queue
from functools import partial
from typing import Protocol

import chess

from chipiron.environments.chess.board import BoardChi
from chipiron.utils import seed
from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils.is_dataclass import IsDataclass
from .factory import create_game_player
from .game_player import GamePlayer, game_player_computes_move_on_board_and_send_move_in_queue
from .player_args import PlayerFactoryArgs
from .player_thread import PlayerProcess


# FIXME double definition of Move functions!!
class MoveFunction(Protocol):
    """Represents a callable object that defines a move function.

    This protocol is used to define the signature of a move function that can be used by the game engine.

    Args:
        board (BoardChi): The game board on which the move function will be applied.
        seed_ (seed): The seed value for the move function.

    Returns:
        None: This function does not return any value.
    """
    def __call__(
            self,
            board: BoardChi,
            seed_: seed
    ) -> None: ...


def send_board_to_player_process_mailbox(
        board: BoardChi,
        seed_: int,
        player_process_mailbox: queue.Queue[BoardMessage]
) -> None:
    """Sends the board and seed to the player process mailbox.

    This function creates a BoardMessage object with the given board and seed,
    and puts it into the player_process_mailbox.

    Args:
        board (BoardChi): The board to send.
        seed_ (int): The seed to send.
        player_process_mailbox (queue.Queue[BoardMessage]): The mailbox to put the message into.
    """
    message: BoardMessage = BoardMessage(
        board=board,
        seed=seed_
    )
    player_process_mailbox.put(item=message)


def create_player_observer(
        player_factory_args: PlayerFactoryArgs,
        player_color: chess.Color,
        distributed_players: bool,
        main_thread_mailbox: queue.Queue[IsDataclass]
) -> tuple[GamePlayer | PlayerProcess, MoveFunction]:
    """Create a player observer.

    This function creates a player observer based on the given parameters. The player observer can be either a `GamePlayer` or a `PlayerProcess`, depending on the value of `distributed_players`.

    Args:
        player_factory_args (PlayerFactoryArgs): The arguments for creating the player.
        player_color (chess.Color): The color of the player.
        distributed_players (bool): A flag indicating whether the players are distributed across multiple processes.
        main_thread_mailbox (queue.Queue[IsDataclass]): The mailbox for communication between the main thread and the player.

    Returns:
        tuple[GamePlayer | PlayerProcess, MoveFunction]: A tuple containing the player observer and the move function.

    """
    generic_player: GamePlayer | PlayerProcess
    move_function: MoveFunction

    # case with multiprocessing when each player is a separate process
    if distributed_players:
        # creating objects Queue that is the mailbox for the player thread
        player_process_mailbox = multiprocessing.Manager().Queue()

        # creating and starting the thread for the player
        player_process: PlayerProcess = PlayerProcess(
            player_factory_args=player_factory_args,
            player_color=player_color,
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
        generic_player = create_game_player(
            player_factory_args=player_factory_args,
            player_color=player_color
        )
        move_function = partial(
            game_player_computes_move_on_board_and_send_move_in_queue,
            game_player=generic_player,
            queue_move=main_thread_mailbox
        )

    return generic_player, move_function
