import threading
import random
import numpy as np
import torch
from datetime import datetime
import time


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


SEED_FIXING_TIMES = [NEVER_FIX_SEED, SEED_FIXED_BEGINNING, SEED_FIXED_EVERY_MOVE] = range(3)
SEED_FIXING_TYPE = [NO_FIX_SEED, FIX_SEED_WITH_CONSTANT, FIX_SEED_WITH_TIME] = range(3)

deterministic_behavior = False
deterministic_mode = SEED_FIXED_EVERY_MOVE
seed_fixing_type = FIX_SEED_WITH_CONSTANT
profiling_bool = True
testing_bool = False
global_lock = Locky()


def init():
    if deterministic_behavior:
        if seed_fixing_type == FIX_SEED_WITH_CONSTANT:
            seed = 11
        elif seed_fixing_type == FIX_SEED_WITH_TIME:
            # Random number with system time
            seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print('seed ', seed)
