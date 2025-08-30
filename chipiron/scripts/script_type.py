"""
This module defines the ScriptType enum and provides a mapping between script types and their corresponding script classes.
"""

from enum import Enum


class ScriptType(str, Enum):
    """
    Enum class representing different types of scripts.
    """

    ONE_MATCH = "one_match"
    LEAGUE = "league"
    LEARN_NN = "learn_nn"
    LEARN_NN_FROM_SCRATCH = "learn_nn_from_scratch"
    BASE_TREE_EXPLORATION = "base_tree_exploration"
    TREE_VISUALIZATION = "tree_visualization"
    REPLAY_MATCH = "replay_match"
