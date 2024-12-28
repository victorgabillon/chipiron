"""
This script defines a graphical user interface (GUI) for interacting with the chipiron program.
The GUI allows the user to select various options and perform different actions such as playing against the chipiron AI, watching a game, or visualizing a tree.

The script_gui function creates the GUI window and handles the user interactions.
The play_against_chipiron, watch_a_game, and visualize_a_tree functions are called when the corresponding buttons are clicked.

Note: This script requires the customtkinter and chipiron modules to be installed.
"""

from typing import Any

import customtkinter as ctk

from chipiron import scripts
from chipiron.players.player_ids import PlayerConfigFile


# TODO switch to pygame

def script_gui() -> tuple[scripts.ScriptType, dict[str, Any]]:
    """
    Creates a graphical user interface (GUI) for interacting with the chipiron program.

    Returns:
        A tuple containing the script type and the arguments for the selected action.
    """
    root = ctk.CTk()
    output: dict[str, Any] = {}
    # place a label on the root window
    root.title('chipiron')

    # frm = ctk.CTkFrame(root, padding=10)

    window_width: int = 800
    window_height: int = 300

    # get the screen dimension
    screen_width: int = root.winfo_screenwidth()
    screen_height: int = root.winfo_screenheight()

    # find the center point
    center_x: int = int(screen_width / 2 - window_width / 2)
    center_y: int = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    # root.iconbitmap('download.jpeg')

    message = ctk.CTkLabel(root, text="What to do?")
    message.grid(column=0, row=0)

    # exit button
    exit_button = ctk.CTkButton(
        root,
        text='Exit',
        command=lambda: root.quit()
    )

    message = ctk.CTkLabel(root, text="Play ")
    message.grid(column=0, row=2)

    # Create the list of options
    color_options_list: list[str] = ["White", "Black"]

    # Variable to keep track of the option
    # selected in OptionMenu
    # color_choice_human = ctk.StringVar(root)
    color_choice_human = ctk.StringVar(value="White")  # set initial value

    # Set the default value of the variable
    # color_choice_human.set("White")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = ctk.CTkOptionMenu(
        master=root,
        values=color_options_list,
        variable=color_choice_human
    )
    strength_menu.grid(column=1, row=2)

    message = ctk.CTkLabel(root, text=" against ")
    message.grid(column=2, row=2)

    # Create the list of options
    chipi_algo_options_list: list[PlayerConfigFile] = [PlayerConfigFile.RecurZipfBase3,
                                                       PlayerConfigFile.Uniform, PlayerConfigFile.Sequool]

    # Variable to keep track of the option
    # selected in OptionMenu
    chipi_algo_choice = ctk.StringVar(value=PlayerConfigFile.RecurZipfBase3)  # set initial value

    # chipi_algo_choice = tk.StringVar(root)

    # Set the default value of the variable
    # chipi_algo_choice.set("RecurZipfBase3")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = ctk.CTkOptionMenu(master=root,
                                      values=chipi_algo_options_list,
                                      variable=chipi_algo_choice)
    strength_menu.grid(column=3, row=2)

    message = ctk.CTkLabel(root, text="  strength: ")

    message.grid(column=5, row=2, padx=10, pady=10)

    # Create the list of options
    options_list = ["1", "2", "3", "4", "5"]

    # Variable to keep track of the option
    # selected in OptionMenu
    strength_value = ctk.StringVar(value="1")  # set initial value

    # strength_value = tk.StringVar(root)

    # Set the default value of the variable
    # strength_value.set("1")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = ctk.CTkOptionMenu(
        master=root,
        variable=strength_value,
        values=options_list
    )
    strength_menu.grid(column=5, row=2, padx=10, pady=10)

    # play button
    play_against_chipiron_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text='!Play!',
        command=lambda: [
            play_against_chipiron(
                output,
                strength=strength_value,
                color=color_choice_human,
                chipi_algo=chipi_algo_choice
            ),
            root.destroy()
        ]
    )

    # watch button
    watch_a_game_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text='Watch a game',
        command=lambda: [watch_a_game(output), root.destroy()]
    )

    # visualize button
    visualize_a_tree_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text='Visualize a tree',
        command=lambda: [visualize_a_tree(output), root.destroy()]
    )

    play_against_chipiron_button.grid(row=2, column=6, padx=10, pady=10)
    watch_a_game_button.grid(row=4, column=0, padx=10, pady=10)
    visualize_a_tree_button.grid(row=6, column=0, padx=10, pady=10)
    exit_button.grid(row=8, column=0, padx=10, pady=10)

    root.mainloop()
    gui_args: dict[str, Any]
    script_type: scripts.ScriptType
    # TODO should this be dict or args or diretly the right dataclass bu then we might need to change abit the parser init logic
    match output['type']:
        case 'play_against_chipiron':
            tree_move_limit = 4 * 10 ** output['strength']
            gui_args = {
                'config_file_name': 'chipiron/scripts/one_match/inputs/base/exp_options.yaml',
                'base_script_args': {
                    'profiling': False
                },
                'match_args': {
                    'file_name_match_setting': 'setting_duda.yaml',
                    'seed': 0
                },
                'gui': True,
            }
            if output['color_human'] == 'White':
                gui_args['match_args']['file_name_player_one'] = PlayerConfigFile.GuiHuman
                gui_args['match_args']['file_name_player_two'] = f'{output["chipi_algo"]}'
                gui_args['match_args']['player_two'] = {
                    'main_move_selector': {'stopping_criterion': {'tree_move_limit': tree_move_limit}}}
            else:
                gui_args['match_args']['file_name_player_two'] = PlayerConfigFile.GuiHuman
                gui_args['match_args']['file_name_player_one'] = f'{output["chipi_algo"]}'
                gui_args['match_args']['player_one'] = {
                    'main_move_selector': {'stopping_criterion': {'tree_move_limit': tree_move_limit}}}
            script_type = scripts.ScriptType.OneMatch
        case 'watch_a_game':
            gui_args = {
                'config_file_name': 'chipiron/scripts/one_match/inputs/base/exp_options.yaml',
                'match_args': {
                    'seed': 0,
                    'gui': True,
                    'file_name_player_one': PlayerConfigFile.RecurZipfBase3,
                    'file_name_player_two': PlayerConfigFile.RecurZipfBase3,
                    'file_name_match_setting': 'setting_duda.yaml'
                }
            }
            script_type = scripts.ScriptType.OneMatch
        case 'tree_visualization':
            gui_args = {
                'config_file_name': 'scripts/tree_visualization/inputs/base/exp_options.yaml',
            }
            script_type = scripts.ScriptType.TreeVisualization
        case other:
            raise Exception(f'Not a good name: {other}')

    print(f'Gui choices: the script name is {script_type} and the args are {gui_args}')
    return script_type, gui_args


def play_against_chipiron(
        output: dict[str, Any],
        strength: ctk.StringVar,
        color: ctk.StringVar,
        chipi_algo: ctk.StringVar
) -> bool:
    """
    Callback function for the "Play" button.
    Sets the output dictionary with the selected options for playing against the chipiron AI.

    Args:
        output: The output dictionary to store the selected options.
        strength: The selected strength level.
        color: The selected color.
        chipi_algo: The selected chipiron algorithm.

    Returns:
        True.
    """
    output['type'] = 'play_against_chipiron'
    output['strength'] = int(strength.get())
    output['color_human'] = str(color.get())
    output['chipi_algo'] = str(chipi_algo.get())
    return True


def watch_a_game(output: dict[str, Any]) -> bool:
    """
    Callback function for the "Watch a game" button.
    Sets the output dictionary with the selected options for watching a game.

    Args:
        output: The output dictionary to store the selected options.

    Returns:
        True.
    """
    output['type'] = 'watch_a_game'
    return True


def visualize_a_tree(output: dict[str, Any]) -> bool:
    """
    Callback function for the "Visualize a tree" button.
    Sets the output dictionary with the selected options for tree visualization.

    Args:
        output: The output dictionary to store the selected options.

    Returns:
        True.
    """
    output['type'] = 'tree_visualization'
    return True
