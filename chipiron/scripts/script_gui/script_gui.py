"""
This module contains a GUI script for the chipiron application.
"""

import tkinter as tk
from typing import Any

from chipiron import scripts


def destroy(root: tk.Tk) -> bool:
    """
    Destroy the root window.

    Args:
        root: The root window to be destroyed.

    Returns:
        bool: True if the root window is destroyed successfully.
    """
    root.destroy()
    return True


# TODO switch to pygame


def script_gui() -> tuple[scripts.ScriptType, dict[str, Any]]:
    """
    Run the chipiron GUI script.

    Returns:
        tuple[scripts.ScriptType, dict[str, Any]]: A tuple containing the script type and the GUI arguments.
    """
    root = tk.Tk()
    # place a label on the root window
    root.title("chipiron")
    output: dict[str, Any] = {}

    window_width: int = 800
    window_height: int = 300

    # get the screen dimension
    screen_width: int = root.winfo_screenwidth()
    screen_height: int = root.winfo_screenheight()

    # find the center point
    center_x: int = int(screen_width / 2 - window_width / 2)
    center_y: int = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    # root.iconbitmap('download.jpeg')

    message = tk.Label(root, text="What to do?")
    message.grid(column=0, row=0)

    # exit button
    exit_button = tk.Button(root, text="Exit", command=lambda: root.quit())

    message = tk.Label(root, text="Play ")
    message.grid(column=0, row=2)

    # Create the list of options
    color_options_list: list[str] = ["White", "Black"]

    # Variable to keep track of the option
    # selected in OptionMenu
    color_choice_human = tk.StringVar(root)

    # Set the default value of the variable
    color_choice_human.set("White")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = tk.OptionMenu(root, color_choice_human, *color_options_list)
    strength_menu.grid(column=1, row=2)

    message = tk.Label(root, text=" against ")
    message.grid(column=2, row=2)

    # Create the list of options
    chipi_algo_options_list: list[str] = ["RecurZipfBase3", "Uniform", "Sequool"]

    # Variable to keep track of the option
    # selected in OptionMenu
    chipi_algo_choice = tk.StringVar(root)

    # Set the default value of the variable
    chipi_algo_choice.set("RecurZipfBase3")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = tk.OptionMenu(root, chipi_algo_choice, *chipi_algo_options_list)
    strength_menu.grid(column=1, row=2)

    message = tk.Label(root, text="  strength: ")

    message.grid(column=4, row=2)

    # Create the list of options
    options_list = ["1", "2", "3", "4", "5"]

    # Variable to keep track of the option
    # selected in OptionMenu
    strength_value = tk.StringVar(root)

    # Set the default value of the variable
    strength_value.set("1")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = tk.OptionMenu(root, strength_value, *options_list)
    strength_menu.grid(column=5, row=2)

    # play button
    play_against_chipiron_button: tk.Button = tk.Button(
        root,
        text="!Play!",
        command=lambda: [
            play_against_chipiron(
                root,
                strength=strength_value,
                color=color_choice_human,
                chipi_algo=chipi_algo_choice,
            ),
            destroy(root),
        ],
    )

    # watch button
    watch_a_game_button: tk.Button = tk.Button(
        root, text="Watch a game", command=lambda: [watch_a_game(output), destroy(root)]
    )

    # visualize button
    visualize_a_tree_button: tk.Button = tk.Button(
        root,
        text="Visualize a tree",
        command=lambda: [visualize_a_tree(output), destroy(root)],
    )

    play_against_chipiron_button.grid(row=2, column=6)
    watch_a_game_button.grid(row=4, column=0)
    visualize_a_tree_button.grid(row=6, column=0)
    exit_button.grid(row=8, column=0)

    root.mainloop()
    gui_args: dict[str, Any]
    script_type: scripts.ScriptType
    match output["type"]:
        case "play_against_chipiron":
            tree_move_limit = 4 * 10 ** output["strength"]
            gui_args = {
                "config_file_name": "chipiron/scripts/one_match/exp_options.yaml",
                "seed": 0,
                "gui": True,
                "file_name_match_setting": "setting_duda.yaml",
            }
            if output["color_human"] == "White":
                gui_args["file_name_player_one"] = "Human.yaml"
                gui_args["file_name_player_two"] = f'{output["chipi_algo"]}.yaml'
                gui_args["player_two"] = {
                    "main_move_selector": {
                        "stopping_criterion": {"tree_move_limit": tree_move_limit}
                    }
                }
            else:
                gui_args["file_name_player_two"] = "Human.yaml"
                gui_args["file_name_player_one"] = f'{output["chipi_algo"]}.yaml'
                gui_args["player_one"] = {
                    "main_move_selector": {
                        "stopping_criterion": {"tree_move_limit": tree_move_limit}
                    }
                }
            script_type = scripts.ScriptType.OneMatch
        case "watch_a_game":
            gui_args = {
                "config_file_name": "chipiron/scripts/one_match/exp_options.yaml",
                "seed": 0,
                "gui": True,
                "file_name_player_one": "RecurZipfBase3.yaml",
                "file_name_player_two": "RecurZipfBase4.yaml",
                "file_name_match_setting": "setting_duda.yaml",
            }
            script_type = scripts.ScriptType.OneMatch
        case "tree_visualization":
            gui_args = {
                "config_file_name": "chipiron/scripts/tree_visualization/exp_options.yaml",
            }
            script_type = scripts.ScriptType.TreeVisualization
        case other:
            raise Exception(f"Not a good name: {other}")

    print(f"Gui choices: the script name is {script_type} and the args are {gui_args}")
    return script_type, gui_args


def play_against_chipiron(
    output: tk.Tk, strength: tk.StringVar, color: tk.StringVar, chipi_algo: tk.StringVar
) -> bool:
    """
    Set the output dictionary to indicate playing against chipiron.

    Args:
        output: The output dictionary to be modified.
        strength: The strength of the player.
        color: The color of the human player.
        chipi_algo: The algorithm choice for chipiron.

    Returns:
        bool: True if the output dictionary is modified successfully.
    """
    output["type"] = "play_against_chipiron"
    output["strength"] = int(strength.get())
    output["color_human"] = str(color.get())
    output["chipi_algo"] = str(chipi_algo.get())
    return True


def watch_a_game(output: dict[str, Any]) -> bool:
    """
    Set the output dictionary to indicate watching a game.

    Args:
        output: The output dictionary to be modified.

    Returns:
        bool: True if the output dictionary is modified successfully.
    """
    output["type"] = "watch_a_game"
    return True


def visualize_a_tree(output: dict[str, Any]) -> bool:
    """
    Set the output dictionary to indicate visualizing a tree.

    Args:
        output: The output dictionary to be modified.

    Returns:
        bool: True if the output dictionary is modified successfully.
    """
    output["type"] = "tree_visualization"
    return True
