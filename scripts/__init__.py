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
    if script_name == 'one_match':
        script_object = OneMatchScript(gui_args)
    elif script_name == 'visualize_tree':
        script_object = VisualizeTreeScript()
    elif script_name == 'learn_nn':
        script_object = LearnNNScript()
    elif script_name == 'record_states':
        script_object = RecordStates()
    elif script_name == 'record_state_eval_stockfish':
        script_object = RecordStateEvalStockfish1()
    elif script_name == 'replay_game':
        script_object = ReplayGameScript()
    elif script_name == 'run_the_league':
        script_object = RuntheLeagueScript()
    else:
        raise Exception(' cannot find ', script_name)
    return script_object
