import copy
import multiprocessing
import queue


# A class that extends the Thread class
class PlayerProcess(multiprocessing.Process):
    def __init__(self, game_player, queue_board, queue_move):
        # Call the Thread class's init function
        multiprocessing.Process.__init__(self, daemon=False)
        self._stop_event = multiprocessing.Event()
        self.game_player = game_player
        self.queue_move = queue_move
        self.queue_board = queue_board

    # Override the run() function of Thread class
    def run(self):
        print('Started player thread : ', self.game_player)

        while not self.stopped():
            import time
            try:
                # print('is there SOMETHING?')

                message = self.queue_board.get(False)
            except queue.Empty:
                #  print('Empty')
                pass
            else:
                # print('OOOOOOOOOOO SOMETHING')

                # Handle task here and call q.task_done()
                if message['type'] == 'board':
                    board = message['board']
                    print('player thread got ', board)
                    if board.turn == self.game_player.color:
                        move = self.game_player.select_move(board)
                        message = {'type': 'move', 'move': move, 'corresponding_board': board.fen()}
                        deep_copy_message = copy.deepcopy(message)
                        print('sending ', message)
                        self.queue_move.put(deep_copy_message)

            # TODO here give option to continue working while the other is thinking

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()