"""
Launching the main chipiron
"""
import argparse
import sys
import scripts


def get_script_and_args(
        raw_command_line_arguments: list
) -> tuple[scripts.ScriptType, dict]:
    """

    Args:
        raw_command_line_arguments: the list of arguments of the scripts given by command line

    Returns:
        A string for the name of script and a dictionary of parameters

    """
    script_type: scripts.ScriptType
    gui_args: dict | None
    # Whether command line arguments are provided or not we ask for more info through a GUI
    if len(raw_command_line_arguments) == 1:  # No args provided
        # use a gui to get user input
        script_type, gui_args = scripts.script_gui()
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

        script_type_str: str = args_command_line['script_name']
        script_type = scripts.ScriptType(script_type_str)
        gui_args = None
    return script_type, gui_args


def main() -> None:
    """
        The main function
    """
    # Getting the command line arguments from the system
    raw_command_line_arguments: list = sys.argv

    script_type: scripts.ScriptType
    gui_args: dict
    # extracting the script_name and possibly some gui input arguments
    script_type, gui_args = get_script_and_args(raw_command_line_arguments)

    # creating the script object from its name and arguments
    script_object: scripts.Script = scripts.create_script(script_type=script_type,
                                                          gui_args=gui_args)

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    main()
