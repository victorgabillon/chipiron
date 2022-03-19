import sys
from scripts.one_match.one_match import OneMatchScript
from scripts.replay_game import ReplayGameScript
from scripts.tree_visualizer import VisualizeTreeScript
from scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import LearnNNScript
from scripts.record_states import RecordStates
from scripts.record_states_eval_stockfish_1 import RecordStateEvalStockfish1
from scripts.runtheleague import RuntheLeagueScript
import tkinter as tk
import argparse


def main():
    # Getting the command line arguments from the system
    raw_command_line_arguments = sys.argv

    # Whether command line arguments are provided or not we ask for more info through a GUI
    if len(raw_command_line_arguments) == 1:  # No args provided
        # use a gui to get user input
        script, gui_args = script_gui()
    else:
        # Capture  the script argument in the command line arguments
        parser_default = argparse.ArgumentParser()
        parser_default.add_argument('--script', type=str, default=None, help='name of the script')
        args_obj, unknown = parser_default.parse_known_args()
        args_command_line = vars(args_obj)  # converting into dictionary format
        script = args_command_line['script']
        gui_args = None

    print('script, gui_args', script, gui_args)

    # launch the relevant script
    if script == 'one_match':
        script_object = OneMatchScript(gui_args)
    elif script == 'visualize_tree':
        script_object = VisualizeTreeScript()
    elif script == 'learn_nn':
        script_object = LearnNNScript()
    elif script == 'record_states':
        script_object = RecordStates()
    elif script == 'record_state_eval_stockfish':
        script_object = RecordStateEvalStockfish1()
    elif script == 'replay_game':
        script_object = ReplayGameScript()
    elif script == 'run_the_league':
        script_object = RuntheLeagueScript()
    else:
        raise Exception(' cannot find ', script)

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


def script_gui():
    root = tk.Tk()
    root.output = None
    # place a label on the root window
    root.title('chipiron')

    window_width = 800
    window_height = 300

    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    # root.iconbitmap('download.jpeg')

    message = tk.Label(root, text="What to do?")
    message.grid(column=0, row=0)

    # exit button
    exit_button = tk.Button(
        root,
        text='Exit',
        command=lambda: root.quit()
    )

    message = tk.Label(root, text="Play Black against chipiron strength:")
    message.grid(column=0, row=2)

    # Create the list of options
    options_list = ["1", "2", "3", "4"]

    # Variable to keep track of the option
    # selected in OptionMenu
    strength_value = tk.StringVar(root)

    # Set the default value of the variable
    strength_value.set("1")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = tk.OptionMenu(root, strength_value, *options_list)
    strength_menu.grid(column=1, row=2)

    # plat button
    play_against_chipiron_button = tk.Button(
        root,
        text='!Play!',
        command=lambda: [play_against_chipiron(root, strength_value), root.destroy()]
    )

    # exit button
    watch_a_game_button = tk.Button(
        root,
        text='Watch a game',
        command=lambda: [watch_a_game(root), root.destroy()]
    )

    play_against_chipiron_button.grid(row=2, column=3)
    watch_a_game_button.grid(row=4, column=0)
    exit_button.grid(row=6, column=0)

    root.mainloop()
    if root.output['type'] == 'play_against_chipiron':
        tree_move_limit = 4 * 10 ** root.output['strength']
        gui_args = {'config_file_name': 'scripts/one_match/exp_options.yaml',
                    'seed': 0,
                    'gui': True,
                    'file_name_player_one': 'RecurZipfBase3.yaml',
                    'file_name_player_two': 'Human.yaml',
                    'file_name_match_setting': 'setting_duda.yaml',
                    'player_one': {'tree_builder': {'stopping_criterion': {'tree_move_limit': tree_move_limit}}}}
        script = 'one_match'

    if root.output['type'] == 'watch_a_game':
        gui_args = {'config_file_name': 'scripts/one_match/exp_options.yaml',
                    'seed': 0,
                    'gui': True,
                    'file_name_player_one': 'RecurZipfBase3.yaml',
                    'file_name_player_two': 'RecurZipfBase4.yaml',
                    'file_name_match_setting': 'setting_duda.yaml'}
        script = 'one_match'

    return script, gui_args


def play_against_chipiron(root, strength_str):
    root.output = {'type': 'play_against_chipiron', 'strength': int(strength_str.get())}


def watch_a_game(root):
    root.output = {'type': 'watch_a_game'}


if __name__ == "__main__":
    main()
