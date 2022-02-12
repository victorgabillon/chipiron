import sys
from scripts.one_match.one_match import OneMatchScript
from scripts.replay_game import ReplayGameScript
from scripts.tree_visualizer import VisualizeTreeScript
from scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import LearnNNScript
from scripts.record_states import RecordStates
from scripts.record_states_eval_stockfish_1 import RecordStateEvalStockfish1
from scripts.runtheleague import RuntheLeagueScript
import tkinter as tk


def main():
    if len(sys.argv) == 1:
        script, gui_args = script_gui()
        print('script, gui_args', script, gui_args)
    else:
        # capture and remove the first argument
        script = sys.argv[1:][0]
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        gui_args = None

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

    window_width = 600
    window_height = 500

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
    message.pack()

    # exit button
    exit_button = tk.Button(
        root,
        text='Exit',
        command=lambda: root.quit()
    )

    # exit button
    play_against_chipiron_button = tk.Button(
        root,
        text='Play against_chipiron as Black',
        command=lambda: [play_against_chipiron(root), root.destroy()]
    )

    # exit button
    watch_a_game_button = tk.Button(
        root,
        text='Watch a game',
        command=lambda: [watch_a_game(root), root.destroy()]
    )

    play_against_chipiron_button.pack(
        ipadx=5,
        ipady=5,
        expand=True
    )

    watch_a_game_button.pack(
        ipadx=5,
        ipady=5,
        expand=True
    )

    exit_button.pack(
        ipadx=5,
        ipady=5,
        expand=True
    )

    root.mainloop()
    if root.output == 'play_against_chipiron':
        gui_args = {'config_file_name': 'chipiron/scripts/one_match/exp_options.yaml',
                    'seed': 0,
                    'gui': True,
                    'file_name_player_one': 'RecurZipfBase3.yaml',
                    'file_name_player_two': 'Human.yaml',
                    'file_name_match_setting': 'setting_duda.yaml'}
        script = 'one_match'

    if root.output == 'watch_a_game':
        gui_args = {'config_file_name': 'chipiron/scripts/one_match/exp_options.yaml',
                    'seed': 0,
                    'gui': True,
                    'file_name_player_one': 'RecurZipfBase3.yaml',
                    'file_name_player_two': 'RecurZipfBase4.yaml',
                    'file_name_match_setting': 'setting_duda.yaml'}
        script = 'one_match'

    return script, gui_args


def play_against_chipiron(root):
    root.output = 'play_against_chipiron'


def watch_a_game(root):
    root.output = 'watch_a_game'


if __name__ == "__main__":
    main()
