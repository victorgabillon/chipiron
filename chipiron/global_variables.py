import threading


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


global_lock = Locky()
