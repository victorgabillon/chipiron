"""
This script defines a graphical user interface (GUI) for interacting with the chipiron program.
The GUI allows the user to select various options and perform different actions such as playing against the chipiron AI, watching a game, or visualizing a tree.

The script_gui function creates the GUI window and handles the user interactions.
The play_against_chipiron, watch_a_game, and visualize_a_tree functions are called when the corresponding buttons are clicked.

Note: This script requires the customtkinter and chipiron modules to be installed.
"""

# TODO switch to pygame
from typing import Any

import customtkinter as ctk
from parsley_coco import make_partial_dataclass_with_optional_paths

from chipiron import scripts
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.MatchTag import MatchConfigTag
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.move_selector.treevalue import TreeAndValuePlayerArgs
from chipiron.players.move_selector.treevalue.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeMoveLimitArgs,
)
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.one_match.one_match import MatchScriptArgs
from chipiron.scripts.script_args import BaseScriptArgs


def script_gui() -> tuple[scripts.ScriptType, dict[str, Any], str]:
    """
    Creates a graphical user interface (GUI) for interacting with the chipiron program.

    Returns:
        A tuple containing the script type and the arguments for the selected action.
    """
    root = ctk.CTk()
    output: dict[str, Any] = {}
    # place a label on the root window
    root.title("chipiron")

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
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    # root.iconbitmap('download.jpeg')

    message = ctk.CTkLabel(root, text="What to do?")
    message.grid(column=0, row=0)

    # exit button
    exit_button = ctk.CTkButton(root, text="Exit", command=lambda: root.quit())

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
        master=root, values=color_options_list, variable=color_choice_human
    )
    strength_menu.grid(column=1, row=2)

    message = ctk.CTkLabel(root, text=" against ")
    message.grid(column=2, row=2)

    # Create the list of options
    chipi_algo_options_list: list[PlayerConfigTag] = [
        PlayerConfigTag.RECUR_ZIPF_BASE_3,
        PlayerConfigTag.UNIFORM,
        PlayerConfigTag.SEQUOOL,
    ]

    # Variable to keep track of the option
    # selected in OptionMenu
    chipi_algo_choice = ctk.StringVar(
        value=PlayerConfigTag.RECUR_ZIPF_BASE_3
    )  # set initial value

    # chipi_algo_choice = tk.StringVar(root)

    # Set the default value of the variable
    # chipi_algo_choice.set("RecurZipfBase3")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = ctk.CTkOptionMenu(
        master=root, values=chipi_algo_options_list, variable=chipi_algo_choice
    )
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
        master=root, variable=strength_value, values=options_list
    )
    strength_menu.grid(column=5, row=2, padx=10, pady=10)

    # play button
    play_against_chipiron_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text="!Play!",
        command=lambda: [
            play_against_chipiron(
                output,
                strength=strength_value,
                color=color_choice_human,
                chipi_algo=chipi_algo_choice,
            ),
            root.destroy(),
        ],
    )

    # play_two_humans button
    play_two_humans_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text="Play between Two Humans",
        command=lambda: [play_two_humans(output), root.destroy()],
    )

    # watch button
    watch_a_game_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text="Watch a game",
        command=lambda: [watch_a_game(output), root.destroy()],
    )

    # visualize button
    visualize_a_tree_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text="Visualize a tree",
        command=lambda: [visualize_a_tree(output), root.destroy()],
    )

    play_against_chipiron_button.grid(row=2, column=6, padx=10, pady=10)
    play_two_humans_button.grid(row=4, column=0, padx=10, pady=10)
    watch_a_game_button.grid(row=6, column=0, padx=10, pady=10)
    visualize_a_tree_button.grid(row=8, column=0, padx=10, pady=10)
    exit_button.grid(row=10, column=0, padx=10, pady=10)

    root.mainloop()

    return generate_inputs(output=output)


def generate_inputs(
    output: dict[str, Any],
) -> tuple[scripts.ScriptType, dict[str, Any], str]:
    script_type: scripts.ScriptType

    PartialOpMatchScriptArgs = make_partial_dataclass_with_optional_paths(
        cls=MatchScriptArgs
    )
    PartialOpMatchArgs = make_partial_dataclass_with_optional_paths(cls=MatchArgs)
    PartialOpPlayerArgs = make_partial_dataclass_with_optional_paths(cls=PlayerArgs)

    PartialOpBaseScriptArgs = make_partial_dataclass_with_optional_paths(
        cls=BaseScriptArgs
    )

    PartialOpTreeAndValuePlayerArgs = make_partial_dataclass_with_optional_paths(
        cls=TreeAndValuePlayerArgs
    )
    PartialOpTreeMoveLimitArgs = make_partial_dataclass_with_optional_paths(
        cls=TreeMoveLimitArgs
    )

    config_file_name: str
    match output["type"]:
        case "play_against_chipiron":
            tree_move_limit = 4 * 10 ** output["strength"]
            gui_args = PartialOpMatchScriptArgs(
                gui=True,
                base_script_args=PartialOpBaseScriptArgs(profiling=False, seed=0),
                match_args=PartialOpMatchArgs(match_setting=MatchConfigTag.Duda),
            )
            config_file_name = "chipiron/scripts/one_match/inputs/human_play_against_computer/exp_options.yaml"

            if output["color_human"] == "White":
                gui_args.match_args.player_one = PlayerConfigTag.GUI_HUMAN
                gui_args.match_args.player_two = PlayerConfigTag(output["chipi_algo"])
                gui_args.match_args.player_two_overwrite = PartialOpPlayerArgs(
                    main_move_selector=PartialOpTreeAndValuePlayerArgs(
                        type=MoveSelectorTypes.TreeAndValue,
                        stopping_criterion=PartialOpTreeMoveLimitArgs(
                            type=StoppingCriterionTypes.TreeMoveLimit,
                            tree_move_limit=tree_move_limit,
                        ),
                    )
                )
            else:
                gui_args.match_args.player_one = PlayerConfigTag(output["chipi_algo"])
                gui_args.match_args.player_two = PlayerConfigTag.GUI_HUMAN
                gui_args.match_args.player_one_overwrite = PartialOpPlayerArgs(
                    main_move_selector=PartialOpTreeAndValuePlayerArgs(
                        stopping_criterion=PartialOpTreeMoveLimitArgs(
                            type=StoppingCriterionTypes.TreeMoveLimit,
                            tree_move_limit=tree_move_limit,
                        )
                    )
                )
            script_type = scripts.ScriptType.OneMatch
        case "play_two_humans":
            config_file_name = "chipiron/scripts/one_match/inputs/human_play_against_human/exp_options.yaml"

            gui_args = PartialOpMatchScriptArgs(
                gui=True,
                base_script_args=PartialOpBaseScriptArgs(profiling=False, seed=0),
                match_args=PartialOpMatchArgs(match_setting=MatchConfigTag.Duda),
            )

            gui_args.match_args.player_one = PlayerConfigTag.GUI_HUMAN
            gui_args.match_args.player_two = PlayerConfigTag.GUI_HUMAN
            script_type = scripts.ScriptType.OneMatch
        case "watch_a_game":
            config_file_name = (
                "chipiron/scripts/one_match/inputs/watch_a_game/exp_options.yaml"
            )
            gui_args = PartialOpMatchScriptArgs(
                gui=True,
                base_script_args=PartialOpBaseScriptArgs(profiling=False, seed=0),
                match_args=PartialOpMatchArgs(
                    match_setting=MatchConfigTag.Duda,
                    player_one=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                    player_two=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                ),
            )

            script_type = scripts.ScriptType.OneMatch
        case "tree_visualization":
            config_file_name = "scripts/tree_visualization/inputs/base/exp_options.yaml"
            gui_args = None
            script_type = scripts.ScriptType.TreeVisualization
        case other:
            raise Exception(f"Not a good name: {other}")

    print(
        f"Gui choices: the script name is {script_type} and the args are {gui_args} and config file {config_file_name}"
    )
    return script_type, gui_args, config_file_name


def play_against_chipiron(
    output: dict[str, Any],
    strength: ctk.StringVar,
    color: ctk.StringVar,
    chipi_algo: ctk.StringVar,
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
    output["type"] = "play_against_chipiron"
    output["strength"] = int(strength.get())
    output["color_human"] = str(color.get())
    output["chipi_algo"] = PlayerConfigTag(str(chipi_algo.get()))
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
    output["type"] = "watch_a_game"
    return True


def play_two_humans(output: dict[str, Any]) -> bool:
    """
    Callback function for the "play_two_humans" button.
    Sets the output dictionary with the selected options for play_two_humans.

    Args:
        output: The output dictionary to store the selected options.

    Returns:
        True.
    """
    output["type"] = "play_two_humans"
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
    output["type"] = "tree_visualization"
    return True
