"""
player_thread.py
"""

import multiprocessing
import queue

import chess

from chipiron.environments.chess.board.factory import BoardFactory
from chipiron.utils import seed
from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils.dataclass import DataClass, IsDataclass

from ..environments.chess.board.factory import create_board_factory
from ..environments.chess.board.utils import FenPlusHistory
from ..scripts.chipiron_args import ImplementationArgs
from .boardevaluators.table_base.factory import SyzygyFactory, create_syzygy_factory
from .factory import create_game_player
from .game_player import (
    GamePlayer,
    game_player_computes_move_on_board_and_send_move_in_queue,
)
from .player_args import PlayerFactoryArgs


# A class that extends the Thread class
class PlayerProcess(multiprocessing.Process):
    """A class representing a player process.

    This class extends the `multiprocessing.Process` class and is responsible for running a player
     in a separate process.

    Attributes:
        game_player (GamePlayer): The game player object.
        queue_board (queue.Queue[DataClass]): The queue for receiving board messages.
        queue_move (queue.Queue[IsDataclass]): The queue for sending move messages.
        player_color (chess.Color): The color of the player.

    Args:
        player_factory_args (PlayerFactoryArgs): The arguments for creating the game player.
        queue_board (queue.Queue[DataClass]): The queue for receiving board messages.
        queue_move (queue.Queue[IsDataclass]): The queue for sending move messages.
        player_color (chess.Color): The color of the player.
    """

    game_player: GamePlayer
    queue_receiving_board: queue.Queue[DataClass]
    queue_sending_move: queue.Queue[IsDataclass]
    player_color: chess.Color
    board_factory: BoardFactory

    def __init__(
        self,
        player_factory_args: PlayerFactoryArgs,
        queue_receiving_board: queue.Queue[DataClass],
        queue_sending_move: queue.Queue[IsDataclass],
        player_color: chess.Color,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> None:
        """Initialize the PlayerThread object.

        Args:
            player_factory_args (PlayerFactoryArgs): The arguments required to create the player.
            queue_receiving_board (queue.Queue[DataClass]): The queue for receiving board data.
            queue_move (queue.Queue[IsDataclass]): The queue for sending move data.
            player_color (chess.Color): The color of the player.

        """
        # Call the Thread class's init function
        multiprocessing.Process.__init__(self, daemon=False)
        self._stop_event = multiprocessing.Event()
        self.queue_sending_move = queue_sending_move
        self.queue_receiving_board = queue_receiving_board
        self.player_color = player_color

        self.board_factory: BoardFactory = create_board_factory(
            use_rust_boards=implementation_args.use_rust_boards,
            use_board_modification=implementation_args.use_board_modification,
            sort_legal_moves=universal_behavior,
        )

        create_syzygy: SyzygyFactory = create_syzygy_factory(
            use_rust=implementation_args.use_rust_boards
        )

        self.game_player: GamePlayer = create_game_player(
            player_factory_args=player_factory_args,
            player_color=player_color,
            syzygy_table=create_syzygy(),
            queue_progress_player=queue_sending_move,
            implementation_args=implementation_args,
            universal_behavior=universal_behavior,
        )
        assert self.game_player.player is not None

    # Override the run() function of Thread class
    def run(self) -> None:
        """
        Executes the player thread.

        This method is called when the player thread is started. It continuously checks for messages in the message queue
        and handles them accordingly. If a message is a `BoardMessage`, it retrieves the board and seed from the message,
        computes the move for the board using the `game_player`, and sends the move to the move queue. If the message is
        not a `BoardMessage`, it simply prints the message.

        Note: This method runs indefinitely until the thread is stopped externally.

        Returns:
            None
        """
        print("Started player thread:", self.game_player)

        while True:
            try:
                message = self.queue_receiving_board.get(False)
            except queue.Empty:
                pass
            else:
                # Handle task here and call q.task_done()
                if isinstance(message, BoardMessage):
                    board_message: BoardMessage = message
                    fen_plus_moves: FenPlusHistory = board_message.fen_plus_moves
                    seed_: seed | None = board_message.seed
                    print(f"Player thread got the board {fen_plus_moves.current_fen}")
                    assert seed_ is not None

                    # the game_player computes the move for the board and sends the move in the move queue
                    game_player_computes_move_on_board_and_send_move_in_queue(
                        fen_plus_history=fen_plus_moves,
                        game_player=self.game_player,
                        queue_move=self.queue_sending_move,
                        seed_int=seed_,
                    )
                else:
                    print(f"NOT EXPECTING THIS MESSAGE !! :  {message}")

            # TODO here give option to continue working while the other is thinking
