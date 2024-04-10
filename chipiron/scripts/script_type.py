"""
This module defines the ScriptType enum and provides a mapping between script types and their corresponding script classes.
"""

from enum import Enum
from typing import Any

from chipiron.scripts.league.runtheleague import RunTheLeagueScript
from .base_tree_exploration.base_tree_exploration import BaseTreeExplorationScript
from .iscript import IScript
from .learn_nn_supervised.learn_nn_from_supervised_datasets import LearnNNScript
from .one_match.one_match import OneMatchScript
from .replay_game.replay_game import ReplayGameScript
from .tree_visualization.tree_visualizer import VisualizeTreeScript


class ScriptType(str, Enum):
    """
    Enum class representing different types of scripts.
    """
    OneMatch = 'one_match'
    League = 'league'
    LearnNN = 'learn_nn'
    BaseTreeExploration = 'base_tree_exploration'
    TreeVisualization = 'tree_visualization'
    ReplayMatch = 'replay_match'


# script_type_to_script_class_name maps Script Type to the class name of the script to instantiate
script_type_to_script_class_name: dict[ScriptType, Any] = {
    ScriptType.OneMatch: OneMatchScript,
    ScriptType.TreeVisualization: VisualizeTreeScript,
    ScriptType.LearnNN: LearnNNScript,
    ScriptType.ReplayMatch: ReplayGameScript,
    ScriptType.League: RunTheLeagueScript,
    ScriptType.BaseTreeExploration: BaseTreeExplorationScript
}


def get_script_type_from_script_class_name(
        script_type: ScriptType
) -> Any:
    """
    Retrieves the script class name based on the given script type.

    Args:
        script_type (ScriptType): The script type.

    Returns:
        Any: The script class name.

    Raises:
        Exception: If the script type is not found.
    """
    if script_type in script_type_to_script_class_name:
        script_class_name: type[IScript] = script_type_to_script_class_name[script_type]
        return script_class_name
    else:
        raise Exception(f'Cannot find the script type {script_type} in file {__name__}')
