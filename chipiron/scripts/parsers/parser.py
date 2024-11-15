"""
This module contains the definition of the MyParser class, which is responsible for parsing command line arguments
and config file arguments for a script.

Classes:
- MyParser: A class for parsing command line arguments and config file arguments.

"""

import os
import sys
from dataclasses import asdict
from typing import Any

import yaml

from chipiron.utils.small_tools import unflatten


class MyParser:
    """
    A class for parsing command line arguments and config file arguments.

    Attributes:
        parser (Any): The parser object used for parsing command line arguments.
        args_command_line (dict[str, Any] | None): The parsed command line arguments.
        args_config_file (dict[str, Any] | None): The parsed config file arguments.
        merged_args (dict[str, Any] | None): The merged arguments from command line, config file, and extra arguments.
        should_parse_command_line_arguments (bool): Whether to parse command line arguments or not.

    Methods:
        __init__(self, parser: Any, should_parse_command_line_arguments: bool = True) -> None:
            Initialize the MyParser object.
        parse_command_line_arguments(self) -> dict[str, Any]:
            Parse the command line arguments using the parser object.
        parse_config_file_arguments(self, config_file_path: str) -> None:
            Parse the config file arguments from the specified config file.
        parse_arguments(self, base_experiment_output_folder: path, extra_args: dict[str, Any] | None = None) -> dict[str, Any]:
            Parse the command line arguments, config file arguments, and extra arguments.
        log_parser_info(self, output_folder: str) -> None:
            Log the parser information to a file.
    """

    parser: Any
    args_command_line: dict[str, Any] | None
    args_config_file: dict[str, Any] | None
    merged_args: dict[str, Any] | None
    should_parse_command_line_arguments: bool = True
    args_class_name: Any  # type[DataclassInstance]

    def __init__(
            self,
            parser: Any,
            args_class_name: Any,  # type[DataclassInstance]
            should_parse_command_line_arguments: bool = True
    ) -> None:
        """
        Initialize the MyParser object.

        Args:
            parser (Any): The parser object used for parsing command line arguments.
            should_parse_command_line_arguments (bool, optional): Whether to parse command line arguments or not.
                Defaults to True.
        """
        self.parser = parser
        self.should_parse_command_line_arguments = should_parse_command_line_arguments
        self.args_class_name = args_class_name

        # attributes to be set and saved at runtime
        self.args_command_line = None
        self.args_config_file = None
        self.merged_args = None

    def parse_command_line_arguments(
            self
    ) -> dict[str, Any]:
        """
        Parse the command line arguments using the parser object.

        Returns:
            dict[str, Any]: A dictionary containing the parsed command line arguments.
        """
        args_obj, unknown = self.parser.parse_known_args()
        args_command_line = vars(args_obj)  # converting into dictionary format
        args_command_line_without_none: dict[str, Any] = {
            key: value for key, value in args_command_line.items() if value is not None
        }
        print('Here are the command line arguments of the script', args_command_line_without_none, sys.argv)

        args_command_line_without_none_unflatten = unflatten(args_command_line_without_none)

        return args_command_line_without_none_unflatten

    def parse_config_file_arguments(
            self,
            config_file_path: str
    ) -> None:
        """
        Parse the config file arguments from the specified config file.

        Args:
            config_file_path (str): The path to the config file.
        """
        try:
            with open(config_file_path, "r") as exp_options_file:
                try:
                    args_config_file = yaml.safe_load(exp_options_file)
                    args_config_file = {} if args_config_file is None else args_config_file
                    print('Here are the yaml file arguments of the script', args_config_file)
                except yaml.YAMLError as exc:
                    print(exc)
        except IOError:
            raise Exception("Could not read file:", config_file_path)
        self.args_config_file = args_config_file

    def parse_arguments(
            self,
            extra_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Parse the command line arguments, config file arguments, and extra arguments.

        Args:
            extra_args (dict[str, Any], optional): Extra arguments to be merged with the parsed arguments.
            Defaults to None.

        Returns:
            dict[str, Any]: A dictionary containing the merged arguments.
        """
        if extra_args is None:
            extra_args = {}
        assert extra_args is not None

        if self.should_parse_command_line_arguments:
            self.args_command_line = self.parse_command_line_arguments()
        else:
            self.args_command_line = {}

        #  the gui/external input  overwrite  the command line arguments
        #  that will overwrite the config file arguments that will overwrite the default arguments
        first_merged_args = self.args_command_line | extra_args

        # 'config_file_name' is a specific input that can be specified either in extra_args or in the command line
        # and that gives the path to a yaml file containing more args
        config_file_path = None
        if 'config_file_name' in first_merged_args:
            config_file_path = first_merged_args['config_file_name']
        if config_file_path is None:
            try:
                self.args_config_file = asdict(self.args_class_name())
            except Exception:
                raise Exception(
                    'The Args dataclass should have all its attribute'
                    ' have default value to have a default instantiation')
        else:
            self.parse_config_file_arguments(config_file_path)
        assert self.args_config_file is not None

        #  the gui input  overwrite  the command line arguments
        #  that overwrite the config file arguments that overwrite the default arguments
        self.merged_args = self.args_config_file | first_merged_args
        print(
            f'Here are the merged arguments of the script {self.merged_args}\n{self.args_config_file}'
            f'\n{self.args_command_line}\n{extra_args}'
        )

        return self.merged_args

    def log_parser_info(
            self,
            output_folder: str
    ) -> None:
        """
        Log the parser information to a file.

        Args:
            output_folder (str): The output folder where the log file will be saved.
        """
        with open(os.path.join(output_folder, 'inputs_and_parsing/base_script_merge.yaml'), 'w') as base_merge:
            yaml.dump(self.merged_args, base_merge, default_flow_style=False)
