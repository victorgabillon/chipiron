"""
This module defines the ScriptType enum and provides a mapping between script types and their corresponding script classes.
"""

from enum import Enum


class ScriptType(str, Enum):
    """
    Enum class representing different types of scripts.
    """

    OneMatch = "one_match"
    League = "league"
    LearnNN = "learn_nn"
    LearnNNFromScratch = "learn_nn_from_scratch"
    BaseTreeExploration = "base_tree_exploration"
    TreeVisualization = "tree_visualization"
    ReplayMatch = "replay_match"
