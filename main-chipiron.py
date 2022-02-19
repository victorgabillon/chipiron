import sys
import subprocess

if __name__ == "__main__":

    # checking if the version of python is high enough
    assert sys.version_info >= (3, 9), "A version of Python higher than 3.9 is required to run chipiron"

    # launching the real main python script.
    # this allows the to bypass the automatic full interpreter check of python that would raise a syntax error before
    # the assertion above in case of a wrong python version
    subprocess.Popen(('python', 'src/main_chipiron.py') + tuple(sys.argv[1:]))
