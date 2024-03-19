"""
random for chipiron
"""
import random

import numpy as np
import torch


def set_seeds(seed=0):
    """
    Set all the base seeds.
    Args:
        seed: the seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
