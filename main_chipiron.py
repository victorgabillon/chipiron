import sys
import subprocess

if __name__ == "__main__":
    # checking if the version of python is high enough
    message = 'A version of Python higher than 3.9 is required to run chipiron.\n' + \
              ' Try using "python3 main_chipiron.py" instead'

    assert sys.version_info >= (3, 9), message
    # launching the real main python script.
    # this allows the to bypass the automatic full interpreter check of python that would raise a syntax error before
    # the assertion above in case of a wrong python version
    subprocess.Popen(('python3', 'main_chipiron_.py') + tuple(sys.argv[1:]))
