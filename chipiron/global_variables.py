import threading
import random


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


deterministic_behavior = True
profiling_bool = False
testing_bool = False
global_lock = Locky()


def init():
    if deterministic_behavior:
        random.seed(22)
