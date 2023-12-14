"""
Launching the main chipiron
"""
from typing import Union
import argparse
import sys
import scripts


def get_script_and_args(
        raw_command_line_arguments: list
) -> tuple[str, dict]:
    """

    Args:
        raw_command_line_arguments: the list of arguments of the scripts given by command line

    Returns:
        A string for the name of script and a dictionary of parameters

    """
    script_name: str
    gui_args: dict | None
    # Whether command line arguments are provided or not we ask for more info through a GUI
    if len(raw_command_line_arguments) == 1:  # No args provided
        # use a gui to get user input
        script_name, gui_args = scripts.script_gui()
    else:
        # Capture  the script argument in the command line arguments
        parser_default: argparse.ArgumentParser = argparse.ArgumentParser()
        parser_default.add_argument('--script_name',
                                    type=str,
                                    default=None,
                                    help='name of the script')
        args_obj, _ = parser_default.parse_known_args()
        args_command_line = vars(args_obj)  # converting into dictionary format
        if args_command_line['script_name'] is None:
            raise ValueError(
                'Expecting command line arguments of the shape python chipiron.py --script_name **name_of script**')

        script_name = args_command_line['script_name']
        gui_args = None
    return script_name, gui_args


def main() -> None:
    """
        The main function
    """
    # Getting the command line arguments from the system
    raw_command_line_arguments: list = sys.argv

    # extracting the script_name and possibly some gui input arguments
    script_name, gui_args = get_script_and_args(raw_command_line_arguments)

    # creating the script object from its name and arguments
    script_object: scripts.Script = scripts.get_script_from_name(script_name, gui_args)

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    main()
