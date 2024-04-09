import multiprocessing
import queue

import chess

from chipiron.environments.chess.board import BoardChi
from chipiron.utils import seed
from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils.is_dataclass import DataClass, IsDataclass
from .factory import create_game_player
from .game_player import GamePlayer, game_player_computes_move_on_board_and_send_move_in_queue
from .player_args import PlayerFactoryArgs


# A class that extends the Thread class
class PlayerProcess(multiprocessing.Process):
    game_player: GamePlayer
    queue_board: queue.Queue[DataClass]
    queue_move: queue.Queue[IsDataclass]
    player_color: chess.Color

    def __init__(
            self,
            player_factory_args: PlayerFactoryArgs,
            queue_board: queue.Queue[DataClass],
            queue_move: queue.Queue[IsDataclass],
            player_color: chess.Color
    ) -> None:
        # Call the Thread class's init function
        multiprocessing.Process.__init__(self, daemon=False)
        self._stop_event = multiprocessing.Event()
        self.queue_move = queue_move
        self.queue_board = queue_board
        self.player_color = player_color

        self.game_player = create_game_player(
            player_factory_args=player_factory_args,
            player_color=player_color
        )
        print('rrt', player_factory_args)
        assert self.game_player.player is not None

    # Override the run() function of Thread class
    def run(self) -> None:
        print('Started player thread : ', self.game_player)

        while True:

            try:
                message = self.queue_board.get(False)
            except queue.Empty:
                pass
            else:
                # Handle task here and call q.task_done()
                if isinstance(message, BoardMessage):
                    board_message: BoardMessage = message
                    board: BoardChi = board_message.board
                    seed_: seed | None = board_message.seed
                    print('player thread got ', board)
                    assert seed_ is not None

                    # the game_player computes the move for the board and sends the move in the move queue
                    game_player_computes_move_on_board_and_send_move_in_queue(
                        board=board,
                        game_player=self.game_player,
                        queue_move=self.queue_move,
                        seed_=seed_
                    )

                else:
                    print(f'opopopopopopopopopdddddddddddddddddsssssssssss, {message}')

            # TODO here give option to continue working while the other is thinking

# def stop(self): from the thread time
#     self._stop_event.set()

# def stopped(self):
#     return self._stop_event.is_set()
