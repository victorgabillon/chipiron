import sys
import scripts
import argparse


def get_script_and_args(raw_command_line_arguments):
    # Whether command line arguments are provided or not we ask for more info through a GUI
    if len(raw_command_line_arguments) == 1:  # No args provided
        # use a gui to get user input
        script_name, gui_args = scripts.script_gui()
    else:
        # Capture  the script argument in the command line arguments
        parser_default = argparse.ArgumentParser()
        parser_default.add_argument('--script_name', type=str, default=None, help='name of the script')
        args_obj, unknown = parser_default.parse_known_args()
        args_command_line = vars(args_obj)  # converting into dictionary format
        script_name = args_command_line['script_name']
        gui_args = None
    return script_name, gui_args


def main():
    # Getting the command line arguments from the system
    raw_command_line_arguments = sys.argv

    # extracting the script_name and possibly some gui input arguments
    script_name, gui_args = get_script_and_args(raw_command_line_arguments)

    # creating the script object from its name and arguments
    script_object: scripts.Script = scripts.get_script_from_name(script_name, gui_args)

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    main()
