"""
factory for scripts module
"""
from enum import Enum
from typing import Any

from scripts.league.runtheleague import RunTheLeagueScript
from scripts.parsers.parser import create_parser, MyParser
from .base_tree_exploration.base_tree_exploration import BaseTreeExplorationScript
from .iscript import IScript
from .learn_nn_supervised.learn_nn_from_supervised_datasets import LearnNNScript
from .one_match.one_match import OneMatchScript
from .replay_game import ReplayGameScript
from .script import Script
from .tree_visualization.tree_visualizer import VisualizeTreeScript


class ScriptType(Enum):
    OneMatch = 'one_match'
    League = 'league'
    LearnNN = 'learn_nn'
    BaseTreeExploration = 'base_tree_exploration'
    TreeVisualization = 'tree_visualization'
    ReplayMatch = 'replay_match'


# instantiate relevant script
def create_script(
        script_type: ScriptType,
        extra_args: dict[str, Any] | None
) -> IScript:
    """
    create the corresponding script
    Args:
        script_type: name of the script to create
        extra_args:

    Returns:
        object:

    """

    # create the relevant script
    parser: MyParser = create_parser()
    base_script: Script = Script(
        parser=parser,
        extra_args=extra_args
    )

    script_object: IScript
    match script_type:
        case ScriptType.OneMatch:
            script_object = OneMatchScript(base_script=base_script)
        case ScriptType.TreeVisualization:
            script_object = VisualizeTreeScript(base_script=base_script)
        case ScriptType.LearnNN:
            script_object = LearnNNScript(base_script=base_script)
        case ScriptType.ReplayMatch:
            script_object = ReplayGameScript()
        case ScriptType.League:
            script_object = RunTheLeagueScript(base_script=base_script)
        case ScriptType.BaseTreeExploration:
            script_object = BaseTreeExplorationScript(base_script=base_script)
        case other:
            raise Exception(f'Cannot find {other} in file {__name__}')
    return script_object
