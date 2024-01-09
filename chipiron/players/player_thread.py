import multiprocessing
import copy
import queue
from .game_player import GamePlayer, send_board_to_game_player


# A class that extends the Thread class
class PlayerProcess(multiprocessing.Process):
    def __init__(self,
                 game_player: GamePlayer,
                 queue_board: queue.Queue,
                 queue_move: queue.Queue,
                 ):
        # Call the Thread class's init function
        multiprocessing.Process.__init__(self, daemon=False)
        self._stop_event = multiprocessing.Event()
        self.game_player = game_player
        self.queue_move = queue_move
        self.queue_board = queue_board

    # Override the run() function of Thread class
    def run(self):
        print('Started player thread : ', self.game_player)

        while True:

            try:
                message = self.queue_board.get(False)
            except queue.Empty:
                pass
            else:
                # Handle task here and call q.task_done()
                if message['type'] == 'board':
                    board = message['board']
                    print('player thread got ', board)

                    send_board_to_game_player(
                        board=board,
                        game_player=self.game_player,
                        queue_move=self.queue_move)

                else:
                    print('opopopopopopopopopdddddddddddddddddsssssssssss')

            # TODO here give option to continue working while the other is thinking

# def stop(self): from the thread time
#     self._stop_event.set()

# def stopped(self):
#     return self._stop_event.is_set()
