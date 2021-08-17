from scripts.one_match import OneMatchScript
from scripts.replay_game import ReplayGameScript
from scripts.tree_visualizer import VisualizeTreeScript
from scripts.learn_nn_from_supervised_datasets import LearnNNScript
from scripts.record_states import RecordStates
from scripts.record_states_eval_stockfish_1 import RecordStateEvalStockfish1
from scripts.runtheleague import RuntheLeagueScript

import sys
import global_variables

def main():
    global_variables.init()
    # print command line arguments
    script = sys.argv[1:][0]
    if script == 'one_match':
        script_object = OneMatchScript()
    elif script == 'visualize_tree':
        script_object = VisualizeTreeScript()
    elif script == 'learn_nn':
        script_object = LearnNNScript()
    elif script == 'record_states':
        script_object = RecordStates()
    elif script == 'record_state_eval_stockfish':
        script_object = RecordStateEvalStockfish1()
    elif script == 'replay_game':
        script_object = ReplayGameScript()
    elif script == 'run_the_league':
        script_object = RuntheLeagueScript()
    else:
        raise Exception(' cannot find ', script)

    script_object.run()
    script_object.terminate()



if __name__ == "__main__":
    main()
