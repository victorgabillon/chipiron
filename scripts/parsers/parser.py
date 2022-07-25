import argparse
import yaml
from chipiron.extra_tools.small_tools import dict_alphabetic_str


def create_parser(default_param_dict):
    parser = argparse.ArgumentParser()
    for key, value in default_param_dict.items():
        parser.add_argument('--' + key, type=type(value), default=None, help='type of nn to learn') #TODO help seems wrong
    return MyParser(parser, default_param_dict)


class MyParser:

    def __init__(self, parser, default_param_dict):
        self.parser_no_default = parser # TODO not clear what it is, it always argparse?
        self.default_param_dict = default_param_dict

        # attributes to be set and saved at runtime
        self.args_command_line = None
        self.args_config_file = None
        self.merged_args = None
        print('oooo')

    def parse_command_line_arguments(self):
        args_obj, unknown = self.parser_no_default.parse_known_args()
        args_command_line = vars(args_obj)  # converting into dictionary format
        self.args_command_line = {key: value for key, value in args_command_line.items() if value is not None}
        print('Here are the command line arguments of the script', self.args_command_line)

    def parse_config_file_arguments(self, config_file_path):

        try:
            with open(config_file_path, "r") as exp_options_file:
                try:
                    args_config_file = yaml.safe_load(exp_options_file)
                    args_config_file = {} if args_config_file is None else args_config_file
                    print('Here are the yaml file arguments of the script', args_config_file)
                except yaml.YAMLError as exc:
                    print(exc)
        except IOError: #TODO weird things hapen here depending on where is execute the script and the type of error catched
            print("Could not read file:", config_file_path)
        self.args_config_file = args_config_file

    def parse_arguments(self, gui_args=None):
        if gui_args is None:
            gui_args = {}

        self.parse_command_line_arguments()

        config_file_path = None
        if 'config_file_name' in self.args_command_line:
            config_file_path = self.args_command_line['config_file_name']
        elif 'config_file_name' in self.default_param_dict:
            config_file_path = self.default_param_dict['config_file_name']
        if config_file_path is None:
            self.args_config_file = {}
        else:
            self.parse_config_file_arguments(config_file_path)

        #  the command line arguments overwrite the config file arguments that overwrite the default arguments
        self.merged_args = self.default_param_dict | self.args_config_file | self.args_command_line | gui_args
        print('Here are the merged arguments of the script', self.merged_args)

        try:
            assert (set(self.default_param_dict.keys()) == set(self.merged_args.keys()))
        except AssertionError as error:
            raise Exception(
                'Please have the set of defaults arguments equals the set of given arguments: {} and {}  || diffs {} {}'.format(
                    self.default_param_dict.keys(), self.merged_args.keys(),
                    set(self.default_param_dict.keys()).difference(set(self.merged_args.keys())),
                    set(self.merged_args.keys()).difference(set(self.default_param_dict.keys()))
                )
            ) from error

        return self.merged_args

    def log_parser_info(self, output_folder):
        with open(output_folder + '/parser_output.txt', 'w') as parser_output:
            parser_output.write('This are the logs of the parsing.\n\n')
            parser_output.write(
                'Default parameters are:\n{}\n\n'.format(dict_alphabetic_str(self.default_param_dict))
            )
            parser_output.write(
                'Command line parameters are:\n{}\n\n'.format(dict_alphabetic_str(self.args_command_line))
            )
            parser_output.write(
                'Config file parameters are:\n{}\n\n'.format(dict_alphabetic_str(self.args_config_file))
            )
            parser_output.write('Merged parameters are:\n{}\n\n'.format(dict_alphabetic_str(self.merged_args)))
