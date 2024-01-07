"""
factory for scripts module
"""
from .one_match.one_match import OneMatchScript
from .replay_game import ReplayGameScript
from .tree_visualization.tree_visualizer import VisualizeTreeScript
from .learn_nn_supervised.learn_nn_from_supervised_datasets import LearnNNScript
from .record_states import RecordStates
from .record_states_eval_stockfish_1 import RecordStateEvalStockfish1
from scripts.league.runtheleague import RunTheLeagueScript
from .script import Script
from .base_tree_exploration.base_tree_exploration import BaseTreeExplorationScript
from scripts.parsers.parser import create_parser, MyParser
from enum import Enum


class ScriptType(Enum):
    OneMatch = 'one_match'
    League = 'league'
    LearnNN = 'learn_nn'
    BaseTreeExploration = 'base_tree_exploration'


# instantiate relevant script
def create_script(
        script_type: ScriptType,
        extra_args: dict | None
) -> Script:
    """
    create the corresponding script
    Args:
        script_type: name of the script to create
        extra_args:

    Returns:

    """

    # create the relevant script
    parser: MyParser = create_parser()
    base_script: Script = Script(parser=parser,
                                 extra_args=extra_args)

    match script_type:
        case ScriptType.OneMatch:
            script_object = OneMatchScript(base_script=base_script)
        case 'tree_visualization':
            script_object = VisualizeTreeScript()
        case ScriptType.LearnNN:
            script_object = LearnNNScript(base_script=base_script)
        case 'record_states':
            script_object = RecordStates()
        case 'record_state_eval_stockfish':
            script_object = RecordStateEvalStockfish1()
        case 'replay_game':
            script_object = ReplayGameScript()
        case ScriptType.League:
            script_object = RunTheLeagueScript(base_script=base_script)
        case ScriptType.BaseTreeExploration:
            script_object = BaseTreeExplorationScript(base_script=base_script)
        case other:
            raise Exception(f'Cannot find {other} in file {__name__}')
    return script_object
