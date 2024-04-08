import os
import sys
from datetime import datetime
from typing import Any

import yaml

from chipiron.utils import path


class MyParser:
    parser: Any
    args_command_line: dict[str, Any] | None
    args_config_file: dict[str, Any] | None
    merged_args: dict[str, Any] | None
    should_parse_command_line_arguments: bool = True

    def __init__(
            self,
            parser: Any,
            should_parse_command_line_arguments: bool = True
    ) -> None:
        self.parser = parser
        self.should_parse_command_line_arguments = should_parse_command_line_arguments

        # attributes to be set and saved at runtime
        self.args_command_line = None
        self.args_config_file = None
        self.merged_args = None

    def parse_command_line_arguments(
            self
    ) -> dict[str, Any]:

        args_obj, unknown = self.parser.parse_known_args()
        args_command_line = vars(args_obj)  # converting into dictionary format
        args_command_line_without_none: dict[str, Any] = {
            key: value for key, value in args_command_line.items() if value is not None
        }
        print('Here are the command line arguments of the script', args_command_line_without_none, sys.argv)

        return args_command_line_without_none

    def parse_config_file_arguments(
            self,
            config_file_path: str
    ) -> None:

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
            base_experiment_output_folder: path,
            extra_args: dict[str, Any] | None = None,
    ):

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

        print('rr', self.args_command_line)
        config_file_path = None
        if 'config_file_name' in first_merged_args:
            config_file_path = first_merged_args['config_file_name']

        if config_file_path is None:
            self.args_config_file = {}
        else:
            self.parse_config_file_arguments(config_file_path)

        assert self.args_config_file is not None

        #  the gui input  overwrite  the command line arguments
        #  that overwrite the config file arguments that overwrite the default arguments
        self.merged_args = self.args_config_file | first_merged_args
        print(
            f'Here are the merged arguments of the script {self.merged_args}\n{self.args_config_file}\n{self.args_command_line}\n{extra_args}')

        if 'output_folder' not in self.merged_args:
            now = datetime.now()  # current date and time
            self.merged_args['experiment_output_folder'] = os.path.join(base_experiment_output_folder, now.strftime(
                "%A-%m-%d-%Y--%H:%M:%S:%f"))
        else:
            self.merged_args['experiment_output_folder'] = os.path.join(base_experiment_output_folder, self.merged_args[
                'output_folder'])

        return self.merged_args

    def log_parser_info(
            self,
            output_folder: str
    ):

        with open(os.path.join(output_folder, 'inputs_and_parsing/base_script_merge.yaml'), 'w') as base_merge:
            yaml.dump(self.merged_args, base_merge, default_flow_style=False)
