import threading

global deterministic_behavior
deterministic_behavior = True

global profiling_bool
profiling_bool = False

global testing_bool
testing_bool = False

global learning_nn_bool
learning_nn_bool = False

global nn_to_train
nn_to_train = None

global nn_replay_buffer_manager
nn_replay_buffer_manager = None



class Locky():
    def __init__(self):
        self.lock = threading.Lock()

    def __getstate__(self):
        return -1

    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def locked(self):
        return self.lock.locked()


def init():
    global global_tree_count
    global_tree_count = 10000

    global global_lock
    global_lock = Locky()
