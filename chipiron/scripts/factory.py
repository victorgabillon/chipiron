"""
factory for scripts module
"""
from typing import Any, TypeVar

from chipiron.scripts.parsers.create_parser import create_parser
from chipiron.scripts.parsers.parser import MyParser
from chipiron.scripts.script_type import ScriptType, get_script_type_from_script_class_name
from chipiron.utils.is_dataclass import DataClass
from .iscript import IScript
from .script import Script

T_IScript = TypeVar("T_IScript", covariant=True, bound=IScript)


# instantiate relevant script
def create_script(
        script_type: ScriptType,
        extra_args: dict[str, Any] | None,
        should_parse_command_line_arguments: bool = True
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
    args_class_name: type[DataClass] = script.args_dataclass_name

    # create the relevant script
    parser: MyParser = create_parser(
        args_class_name=args_class_name,
        should_parse_command_line_arguments=should_parse_command_line_arguments
    )

    base_script: Script = Script(
        parser=parser,
        extra_args=extra_args
    )

    script_class_name: type[IScript] = get_script_type_from_script_class_name(script_type=script_type)
    script_object: IScript = script_class_name(base_script=base_script)

    return script_object
