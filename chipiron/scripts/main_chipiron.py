"""Launching the main chipiron."""

import argparse
import logging
import sys
from typing import TYPE_CHECKING, Any

from chipiron.scripts.factory import create_script
from chipiron.scripts.script_args import LoggingArgs
from chipiron.scripts.script_gui.script_gui_custom import script_gui
from chipiron.scripts.script_type import ScriptType
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from chipiron.scripts.iscript import IScript

sys.path.append("../../")

# Configure parsley_coco logging to reduce noise
try:
    from parsley_coco.logger import set_verbosity

    set_verbosity(logging.WARNING)
except ImportError:
    # parsley_coco might not be available in all environments
    pass


def get_script_and_args(
    raw_command_line_arguments: list[str],
) -> tuple[ScriptType, IsDataclass | None, str | None, LoggingArgs]:
    """Args:.

        raw_command_line_arguments: the list of arguments of the scripts given by command line

    Returns:
        A string for the name of script and a dictionary of parameters

    """
    # First capture  the script name if present and the debug level if present
    parser_logging_script_name: argparse.ArgumentParser = argparse.ArgumentParser()
    parser_logging_script_name.add_argument(
        "--base_script_args.logging_levels.chipiron",
        type=int,
        default=None,
        help="logging level for chipiron",
    )
    parser_logging_script_name.add_argument(
        "--base_script_args.logging_levels.parsley",
        type=int,
        default=None,
        help="logging level for parsley",
    )
    parser_logging_script_name.add_argument(
        "--script_name", type=str, default=None, help="name of the script"
    )
    args_obj_logging_script_name, _ = parser_logging_script_name.parse_known_args()
    args_logging_script_name: dict[Any, Any] = vars(args_obj_logging_script_name)

    logging_args: LoggingArgs = LoggingArgs()

    if args_logging_script_name["base_script_args.logging_levels.chipiron"] is not None:
        logging_args.chipiron = args_logging_script_name[
            "base_script_args.logging_levels.chipiron"
        ]

    if args_logging_script_name["base_script_args.logging_levels.parsley"] is not None:
        logging_args.parsley = args_logging_script_name[
            "base_script_args.logging_levels.parsley"
        ]

    # setting the logging levels with the values provided by the user in the command line or gui if given
    # if provided by confifile of defualt internal value or others this will be overwitten latter once the main parser (parsley is run, usually upon script creation)
    chipiron_logger.setLevel(logging_args.chipiron)

    script_type: ScriptType
    extra_args: IsDataclass | None = None
    config_file_name: str | None = None

    # Whether command line arguments are provided or not we ask for more info through a GUI
    if args_logging_script_name["script_name"] is None:  # No script name provided
        # use a gui to get user input
        gui_extra_args: IsDataclass | None
        script_type, gui_extra_args, config_file_name = script_gui()
        extra_args = gui_extra_args
    else:
        # first parse/retrieve the name of the script then look for the names of the parameters related to this script
        # then parse again and retrieve the parameters related to the script if specified

        script_type_str: str = args_logging_script_name["script_name"]
        script_type = ScriptType(script_type_str)

        extra_args = None

    return script_type, extra_args, config_file_name, logging_args


def main() -> None:
    """Run the main function."""
    # Getting the command line arguments from the system
    raw_command_line_arguments: list[str] = sys.argv

    # the type of script to be executed
    script_type: ScriptType

    # arguments provided to the script from the outside. Here it can be from a gui or command line
    extra_args: IsDataclass | None

    logging_args: LoggingArgs

    # extracting the script_name and possibly some input arguments from either the gui or a yaml file or command line
    script_type, extra_args, config_file_name, logging_args = get_script_and_args(
        raw_command_line_arguments
    )

    try:
        from parsley_coco.logger import set_verbosity

        set_verbosity(logging_args.parsley)
    except ImportError:
        # parsley_coco might not be available in all environments
        pass

    # creating the script object from its name and arguments
    script_object: IScript = create_script(
        script_type=script_type,
        extra_args=extra_args,
        config_file_name=config_file_name,
        should_parse_command_line_arguments=True,
        parsley_logging_level=logging_args.parsley,
    )

    # Print chipiron startup banner with chipiron-themed icons
    chipiron_logger.info("=" * 60)
    chipiron_logger.info("              🦑 ♛  CHIPIRON MAIN STARTS  ♛ 🦑")
    chipiron_logger.info("           🐙 AI Chess Engine & Learning System 🐙")
    chipiron_logger.info("      ♔ • Ready to play and learn chess like a pulpo!  •♕")
    chipiron_logger.info("       🦑 ~ Swimming through chess possibilities ~ 🦑")
    chipiron_logger.info("=" * 60)
    print()

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    # checking if the version of python is high enough
    message = (
        "A version of Python higher than 3.13 is required to run chipiron.\n"
        ' Try using "python3 main_chipiron.py" instead'
    )

    assert sys.version_info >= (3, 13), message
    # launching the real main python script.
    # this allows the to bypass the automatic full interpreter check of python that would raise a syntax error before
    # the assertion above in case of a wrong python version
    main()
