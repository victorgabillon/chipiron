from scripts.one_match import OneMatchScript
from scripts.learn_and_classify_nn import LearnAndClassifyScript
from scripts.learn_nn_from_supervised_gameover_datasets import LearnNNScript

import sys


def main():
    # print command line arguments
    script = sys.argv[1:][0]
    if script == 'one_match':
        script_object = OneMatchScript()
    elif script == 'learn_and_classify':
        script_object = LearnAndClassifyScript()
    elif script == 'learn_nn':
        script_object = LearnNNScript()
    else:
        raise Exception(' cannot find ', script)

    script_object.run()


if __name__ == "__main__":
    main()
