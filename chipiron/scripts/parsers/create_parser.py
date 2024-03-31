import argparse
from dataclasses import Field, fields
from typing import Any

from .parser import MyParser


def create_parser(
        args_class_name: Any,  # type[DataclassInstance]
        should_parse_command_line_arguments: bool = True
) -> MyParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    # one can specify a path to a yaml file containing parameters
    # that can be turned into the class named args_class_name (with dacite)
    parser.add_argument(
        '--config_file_name',
        type=str,
        default=None,
        help='path to a yaml file with arguments for the script'

    )

    # one can  specify parameters from the class named args_class_name
    # that will overwrite the ones in the yaml file
    field: Field[Any]
    for field in fields(args_class_name):
        parser.add_argument(
            str('--' + field.name),
            type=str,
            default=None,
            help='to be written'
        )

    my_parser: MyParser = MyParser(
        parser=parser,
        should_parse_command_line_arguments=should_parse_command_line_arguments
    )

    return my_parser
