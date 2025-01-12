"""
random for chipiron
"""

import random

import numpy as np
import torch


def set_seeds(seed: int = 0) -> None:
    """
    Set all the base seeds.

    Args:
        seed(int): the seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
