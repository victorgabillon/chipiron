"""
factory for scripts module
"""

import logging
from typing import Any

from parsley_coco import Parsley, create_parsley

from chipiron.utils.dataclass import DataClass
from chipiron.utils.logger import chipiron_logger

from .get_script import get_script_type_from_script_class_name
from .iscript import IScript
from .script import Script
from .script_type import ScriptType


# instantiate relevant script
def create_script(
    script_type: ScriptType,
    extra_args: dict[str, Any] | None = None,
    config_file_name: str | None = None,
    should_parse_command_line_arguments: bool = True,
) -> IScript:
    """
    Creates the corresponding script

    Args:
        should_parse_command_line_arguments: whether the script should parse the command line. Default is True.
        But might be set to False if called from a higher level script that wants the current script to stick to some
        defaults argument or if the higher level script wants to set them thought the extra_args.

        script_type: name of the script to create

        extra_args: arguments provided to the script from the outside. In some cases they originate from a gui
         or command line. It can also be any dict specified to this function


    Returns:
        object:

    """

    # retrieve the name of the class of args associated to this script
    script: Any = get_script_type_from_script_class_name(script_type=script_type)
    args_dataclass_name: type[DataClass] = script.args_dataclass_name

    # create the relevant script
    parser: Parsley[Any] = create_parsley(
        args_dataclass_name=args_dataclass_name,
        should_parse_command_line_arguments=should_parse_command_line_arguments,
        logger=chipiron_logger,  # not working at the moment
        verbosity=logging.INFO,
    )

    base_script: Script[Any] = Script(
        parser=parser, extra_args=extra_args, config_file_name=config_file_name
    )

    script_class_name: type[IScript] = get_script_type_from_script_class_name(
        script_type=script_type
    )
    script_object: IScript = script_class_name(base_script=base_script)

    return script_object
