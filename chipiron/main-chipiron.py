from scripts.one_match import OneMatchScript
from scripts.learn_and_classify_nn import LearnAndClassifyScript
from scripts.learn_nn_from_supervised_gameover_datasets import LearnNNScript
from scripts.record_states import RecordStates
import sys
import global_variables


def main():
    global_variables.init()
    # print command line arguments
    script = sys.argv[1:][0]
    if script == 'one_match':
        script_object = OneMatchScript()
    elif script == 'learn_and_classify':
        script_object = LearnAndClassifyScript()
    elif script == 'learn_nn':
        script_object = LearnNNScript()
    elif script == 'record_states':
        script_object = RecordStates()
    else:
        raise Exception(' cannot find ', script)

    script_object.run()
    script_object.terminate()



if __name__ == "__main__":
    main()
