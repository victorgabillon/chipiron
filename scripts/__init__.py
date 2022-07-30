from scripts.one_match.one_match import OneMatchScript
from scripts.replay_game import ReplayGameScript
from scripts.tree_visualizer import VisualizeTreeScript
from scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import LearnNNScript
from scripts.record_states import RecordStates
from scripts.record_states_eval_stockfish_1 import RecordStateEvalStockfish1
from scripts.runtheleague import RuntheLeagueScript
from scripts.script_gui import script_gui
from scripts.script import Script


# instantiate relevant script
def get_script_from_name(script_name: str, gui_args: object) -> Script:
    # launch the relevant script
    match script_name:
        case  'one_match':
            script_object = OneMatchScript(gui_args)
        case 'visualize_tree':
            script_object = VisualizeTreeScript()
        case 'learn_nn':
            script_object = LearnNNScript()
        case 'record_states':
            script_object = RecordStates()
        case 'record_state_eval_stockfish':
            script_object = RecordStateEvalStockfish1()
        case 'replay_game':
            script_object = ReplayGameScript()
        case 'run_the_league':
            script_object = RuntheLeagueScript()
        case other:
            raise Exception(' cannot find ', other)
    return script_object
