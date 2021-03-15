from scripts.one_match import OneMatchScript

import sys


def main():
    # print command line arguments
    script = sys.argv[1:][0]
    if script == 'one_match':
        script_object = OneMatchScript()
    elif script == 'learn_and_classify':
        script_object = OneMatchScript()

    script_object.run()

if __name__ == "__main__":
    main()
