"""
This module provides a function to create an argument parser for command line arguments.

The `create_parser` function takes in the name of a dataclass, which represents the arguments
that can be parsed from the command line. It creates an instance of `argparse.ArgumentParser`
and adds arguments based on the fields of the dataclass.

The function also allows specifying a YAML file that contains additional arguments, which can
be turned into the dataclass using the `dacite` library. These arguments can be overwritten
by specifying them directly on the command line.

The function returns an instance of `MyParser`, which is a custom wrapper around
`argparse.ArgumentParser`. This wrapper provides additional functionality for parsing
and accessing the command line arguments.

Example usage:
    # Create an argument parser for MyArgs dataclass
    parser = create_parser(MyArgs)

    # Parse command line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    print(args.my_argument)
"""

import argparse
from dataclasses import Field, fields
from typing import Any

from .parser import MyParser


def create_parser(
        args_class_name: Any,  # type[DataclassInstance]
        should_parse_command_line_arguments: bool = True
) -> MyParser:
    """
    Create an argument parser for command line arguments.

    Args:
        args_class_name: The name of the dataclass representing the arguments.
        should_parse_command_line_arguments: Whether to parse command line arguments.

    Returns:
        An instance of MyParser, which is a custom wrapper around argparse.ArgumentParser.

    """
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
        args_class_name=args_class_name,
        should_parse_command_line_arguments=should_parse_command_line_arguments
    )

    return my_parser
