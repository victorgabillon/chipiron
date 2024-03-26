"""
Launching the main chipiron
"""
import argparse
import sys
from typing import Any

import scripts
from chipiron.utils.small_tools import yaml_fetch_args_in_file


def get_script_and_args(
        raw_command_line_arguments: list[str]
) -> tuple[scripts.ScriptType, dict[str, Any] | None]:
    """

    Args:
        raw_command_line_arguments: the list of arguments of the scripts given by command line

    Returns:
        A string for the name of script and a dictionary of parameters

    """
    script_type: scripts.ScriptType
    extra_args: dict[str, Any] | None = None
    # Whether command line arguments are provided or not we ask for more info through a GUI
    if len(raw_command_line_arguments) == 1:  # No args provided
        # use a gui to get user input
        gui_extra_args: dict[str, Any] | None
        script_type, gui_extra_args = scripts.script_gui()
        extra_args = gui_extra_args
    else:
        # Capture  the script argument in the command line arguments
        parser_default: argparse.ArgumentParser = argparse.ArgumentParser()
        parser_default.add_argument(
            '--script_name',
            type=str,
            default=None,
            help='name of the script'
        )
        parser_default.add_argument(
            '--config_file_name',
            type=str,
            default=None,
            help='path to a yaml file with arguments for the script'
        )
        args_obj, _ = parser_default.parse_known_args()
        args_command_line = vars(args_obj)  # converting into dictionary format
        if args_command_line['script_name'] is None:
            raise ValueError(
                'Expecting command line arguments of the shape python chipiron.py --script_name **name_of script**')

        script_type_str: str = args_command_line['script_name']
        script_type = scripts.ScriptType(script_type_str)
        if args_command_line['config_file_name'] is not None:
            command_line_extra_args: dict[str, Any] = yaml_fetch_args_in_file(
                path_file=args_command_line['config_file_name'])
            extra_args = command_line_extra_args
    return script_type, extra_args


def main() -> None:
    """
        The main function
    """
    # Getting the command line arguments from the system
    raw_command_line_arguments: list[str] = sys.argv

    script_type: scripts.ScriptType
    extra_args: dict[str, Any] | None
    # extracting the script_name and possibly some input arguments from either the gui or a yaml file
    script_type, extra_args = get_script_and_args(raw_command_line_arguments)

    # creating the script object from its name and arguments
    script_object: scripts.IScript = scripts.create_script(
        script_type=script_type,
        extra_args=extra_args
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    main()
