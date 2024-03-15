import copy
import multiprocessing
import queue


# A class that extends the Thread class
class SyzygyProcess(multiprocessing.Process):
    def __init__(self, syzygy_table, queue_board):
        # Call the Thread class's init function
        multiprocessing.Process.__init__(self, daemon=False)
        self._stop_event = multiprocessing.Event()
        self.syzygy_table = syzygy_table
        self.queue_board = queue_board

    # Override the run() function of Thread class
    def run(self):
        print('Started Syzygy thread : ', self.syzygy_table)

        while not self.stopped():

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
                    queue_reply = message['queue_reply']
                    print('syzygy thread got ', board)
                    move = self.syzygy_table.best_move(board)
                    message = {'type': 'move', 'move': move, 'corresponding_board': board.fen()}
                    deep_copy_message = copy.deepcopy(message)
                    print('sending ', message)
                    queue_reply.put(deep_copy_message)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
