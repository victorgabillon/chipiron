"""
Launching the main chipiron
"""

import argparse
import sys
from typing import Any

from chipiron.scripts.factory import create_script
from chipiron.scripts.iscript import IScript
from chipiron.scripts.script_gui.script_gui_custom import script_gui
from chipiron.scripts.script_type import ScriptType

sys.path.append("../../")


def get_script_and_args(
    raw_command_line_arguments: list[str],
) -> tuple[ScriptType, dict[str, Any] | None, str | None]:
    """

    Args:
        raw_command_line_arguments: the list of arguments of the scripts given by command line

    Returns:
        A string for the name of script and a dictionary of parameters

    """
    script_type: ScriptType
    extra_args: dict[str, Any] | None = None
    config_file_name: str | None = None
    # Whether command line arguments are provided or not we ask for more info through a GUI
    if len(raw_command_line_arguments) == 1:  # No args provided
        # use a gui to get user input
        gui_extra_args: dict[str, Any] | None
        script_type, gui_extra_args, config_file_name = script_gui()
        extra_args = gui_extra_args
    else:
        # first parse/retrieve the name of the script then look for the names of the parameters related to this script
        # then parse again and retrieve the parameters related to the script if specified

        # Capture  the script argument in the command line arguments
        parser_default: argparse.ArgumentParser = argparse.ArgumentParser()
        parser_default.add_argument(
            "--script_name", type=str, default=None, help="name of the script"
        )
        args_obj, _ = parser_default.parse_known_args()
        args_command_line: dict[Any, Any] = vars(
            args_obj
        )  # converting into dictionary format

        # print("command line arguments:", args_command_line)

        # the script name must be specified otherwise fail
        if args_command_line["script_name"] is None:
            raise ValueError(
                "Expecting command line arguments of the shape python chipiron.py --script_name **name_of script**"
            )

        script_type_str: str = args_command_line["script_name"]
        script_type = ScriptType(script_type_str)

        extra_args = {}

    # print("extra_args", extra_args)
    return script_type, extra_args, config_file_name


def main() -> None:
    """
    The main function
    """
    # Getting the command line arguments from the system
    raw_command_line_arguments: list[str] = sys.argv

    # the type of script to be executed
    script_type: ScriptType

    # arguments provided to the script from the outside. Here it can be from a gui or command line
    extra_args: dict[str, Any] | None

    # extracting the script_name and possibly some input arguments from either the gui or a yaml file or command line
    script_type, extra_args, config_file_name = get_script_and_args(
        raw_command_line_arguments
    )

    # creating the script object from its name and arguments
    script_object: IScript = create_script(
        script_type=script_type,
        extra_args=extra_args,
        config_file_name=config_file_name,
        should_parse_command_line_arguments=True,
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    # checking if the version of python is high enough
    message = (
        "A version of Python higher than 3.10 is required to run chipiron.\n"
        + ' Try using "python3 main_chipiron.py" instead'
    )

    assert sys.version_info >= (3, 10), message
    # launching the real main python script.
    # this allows the to bypass the automatic full interpreter check of python that would raise a syntax error before
    # the assertion above in case of a wrong python version
    main()
