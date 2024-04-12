"""
This script is the entry point for running the chipiron application.

Chipiron is a Python application that requires a version of Python higher than 3.10 to run.
If the Python version is not compatible, an assertion error will be raised.

The script launches the main_chipiron2.py script using the subprocess module, allowing it to bypass
the automatic full interpreter check of Python that would raise a syntax error before the assertion check.

Usage: python3 main_chipiron.py [arguments]
"""

import subprocess
import sys

if __name__ == "__main__":
    # checking if the version of python is high enough
    message = 'A version of Python higher than 3.10 is required to run chipiron.\n' + \
              ' Try using "python3 main_chipiron.py" instead'

    assert sys.version_info >= (3, 10), message
    # launching the real main python script.
    # this allows the to bypass the automatic full interpreter check of python that would raise a syntax error before
    # the assertion above in case of a wrong python version
    subprocess.Popen(('python3', 'main_chipiron_.py') + tuple(sys.argv[1:]))
