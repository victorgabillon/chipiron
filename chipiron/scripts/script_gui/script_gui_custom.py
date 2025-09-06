"""
This script defines a graphical user interface (GUI) for interacting with the chipiron program.
The GUI allows the user to select various options and perform different actions such as playing against the chipiron AI, watching a game, or visualizing a tree.

The script_gui function creates the GUI window and handles the user interactions.
The play_against_chipiron, watch_a_game, and visualize_a_tree functions are called when the corresponding buttons are clicked.

Note: This script requires the customtkinter and chipiron modules to be installed.
"""

# TODO switch to pygame
from typing import TYPE_CHECKING, Any, cast

import customtkinter as ctk
from parsley_coco import make_partial_dataclass_with_optional_paths

from chipiron import scripts
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_tag import MatchConfigTag
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
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    import tkinter as tk


def format_gui_args_for_display(gui_args: Any) -> str:
    """
    Format GUI arguments for human-readable display in logging.

    Args:
        gui_args: The GUI arguments object to format

    Returns:
        A formatted string representation of the arguments
    """
    if gui_args is None:
        return "None"

    args_parts: list[str] = []

    if hasattr(gui_args, "gui") and gui_args.gui:
        args_parts.append("GUI: Enabled")

    if hasattr(gui_args, "match_args") and gui_args.match_args:
        match_args = gui_args.match_args
        if hasattr(match_args, "player_one") and match_args.player_one:
            args_parts.append(
                f"Player One: {match_args.player_one.name if hasattr(match_args.player_one, 'name') else match_args.player_one}"
            )
        if hasattr(match_args, "player_two") and match_args.player_two:
            args_parts.append(
                f"Player Two: {match_args.player_two.name if hasattr(match_args.player_two, 'name') else match_args.player_two}"
            )
        if hasattr(match_args, "match_setting") and match_args.match_setting:
            args_parts.append(
                f"Match Setting: {match_args.match_setting.name if hasattr(match_args.match_setting, 'name') else match_args.match_setting}"
            )

        # Display player overwrite settings
        if (
            hasattr(match_args, "player_one_overwrite")
            and match_args.player_one_overwrite
        ):
            overwrite = match_args.player_one_overwrite
            if (
                hasattr(overwrite, "main_move_selector")
                and overwrite.main_move_selector
            ):
                selector = overwrite.main_move_selector
                if (
                    hasattr(selector, "stopping_criterion")
                    and selector.stopping_criterion
                ):
                    criterion = selector.stopping_criterion
                    if hasattr(criterion, "tree_move_limit"):
                        args_parts.append(
                            f"Player One Strength: {criterion.tree_move_limit} moves"
                        )

        if (
            hasattr(match_args, "player_two_overwrite")
            and match_args.player_two_overwrite
        ):
            overwrite = match_args.player_two_overwrite
            if (
                hasattr(overwrite, "main_move_selector")
                and overwrite.main_move_selector
            ):
                selector = overwrite.main_move_selector
                if (
                    hasattr(selector, "stopping_criterion")
                    and selector.stopping_criterion
                ):
                    criterion = selector.stopping_criterion
                    if hasattr(criterion, "tree_move_limit"):
                        args_parts.append(
                            f"Player Two Strength: {criterion.tree_move_limit} moves"
                        )

    if hasattr(gui_args, "base_script_args") and gui_args.base_script_args:
        base_args = gui_args.base_script_args
        if hasattr(base_args, "profiling") and base_args.profiling is not None:
            args_parts.append(
                f"Profiling: {'Enabled' if base_args.profiling else 'Disabled'}"
            )
        if hasattr(base_args, "seed") and base_args.seed is not None:
            args_parts.append(f"Random Seed: {base_args.seed}")

    return "\n    ".join(args_parts) if args_parts else "Default settings"


def script_gui() -> tuple[scripts.ScriptType, IsDataclass | None, str]:
    """
    Creates a graphical user interface (GUI) for interacting with the chipiron program.

    Returns:
        A tuple containing the script type and the arguments for the selected action.
    """
    root: ctk.CTk = ctk.CTk()
    output: dict[str, Any] = {}
    # place a label on the root window
    root.title("🐙 chess with chipirons 🐙")

    # frm = ctk.CTkFrame(root, padding=10)

    window_width: int = 900
    window_height: int = 300

    # get the screen dimension
    screen_width: int = root.winfo_screenwidth()
    screen_height: int = root.winfo_screenheight()

    # find the center point
    center_x: int = int(screen_width / 2 - window_width / 2)
    center_y: int = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    # root.iconbitmap('download.jpg')

    message = cast("tk.Widget", ctk.CTkLabel(root, text="What to do?"))
    message.grid(column=0, row=0)

    # exit button
    exit_button = ctk.CTkButton(root, text="Exit", command=lambda: root.quit())

    message = cast("tk.Widget", ctk.CTkLabel(root, text="Play "))
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
    cast("tk.Widget", strength_menu).grid(column=1, row=2)

    message = ctk.CTkLabel(root, text=" against ")
    cast("tk.Widget", message).grid(column=2, row=2)

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
    cast("tk.Widget", strength_menu).grid(column=3, row=2)

    message = ctk.CTkLabel(root, text="  strength: ")

    cast("tk.Widget", message).grid(column=5, row=2, padx=10, pady=10)

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
    cast("tk.Widget", strength_menu).grid(column=5, row=2, padx=10, pady=10)

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

    cast("tk.Widget", play_against_chipiron_button).grid(
        row=2, column=6, padx=10, pady=10
    )
    cast("tk.Widget", play_two_humans_button).grid(row=4, column=0, padx=10, pady=10)
    cast("tk.Widget", watch_a_game_button).grid(row=6, column=0, padx=10, pady=10)
    cast("tk.Widget", visualize_a_tree_button).grid(row=8, column=0, padx=10, pady=10)
    cast("tk.Widget", exit_button).grid(row=10, column=0, padx=10, pady=10)

    cast("tk.Tk", root).mainloop()

    return generate_inputs(output=output)


def generate_inputs(
    output: dict[str, Any],
) -> tuple[scripts.ScriptType, IsDataclass | None, str]:
    """Generates script type and arguments based on the GUI output.

    Args:
        output (dict[str, Any]): The output from the GUI.

    Raises:
        ValueError: If the output is invalid.

    Returns:
        tuple[scripts.ScriptType, dict[str, Any], str]: The script type, arguments, and config file name.
    """
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
                match_args=PartialOpMatchArgs(match_setting=MatchConfigTag.DUDA),
            )
            config_file_name = "chipiron/scripts/one_match/inputs/human_play_against_computer/exp_options.yaml"

            if output["color_human"] == "White":
                gui_args.match_args.player_one = PlayerConfigTag.GUI_HUMAN
                gui_args.match_args.player_two = PlayerConfigTag(output["chipi_algo"])
                gui_args.match_args.player_two_overwrite = PartialOpPlayerArgs(
                    main_move_selector=PartialOpTreeAndValuePlayerArgs(
                        type=MoveSelectorTypes.TreeAndValue,
                        stopping_criterion=PartialOpTreeMoveLimitArgs(
                            type=StoppingCriterionTypes.TREE_MOVE_LIMIT,
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
                            type=StoppingCriterionTypes.TREE_MOVE_LIMIT,
                            tree_move_limit=tree_move_limit,
                        )
                    )
                )
            script_type = scripts.ScriptType.ONE_MATCH
        case "play_two_humans":
            config_file_name = "chipiron/scripts/one_match/inputs/human_play_against_human/exp_options.yaml"

            gui_args = PartialOpMatchScriptArgs(
                gui=True,
                base_script_args=PartialOpBaseScriptArgs(profiling=False, seed=0),
                match_args=PartialOpMatchArgs(match_setting=MatchConfigTag.DUDA),
            )

            gui_args.match_args.player_one = PlayerConfigTag.GUI_HUMAN
            gui_args.match_args.player_two = PlayerConfigTag.GUI_HUMAN
            script_type = scripts.ScriptType.ONE_MATCH
        case "watch_a_game":
            config_file_name = (
                "chipiron/scripts/one_match/inputs/watch_a_game/exp_options.yaml"
            )
            gui_args = PartialOpMatchScriptArgs(
                gui=True,
                base_script_args=PartialOpBaseScriptArgs(profiling=False, seed=0),
                match_args=PartialOpMatchArgs(
                    match_setting=MatchConfigTag.DUDA,
                    player_one=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                    player_two=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                ),
            )

            script_type = scripts.ScriptType.ONE_MATCH
        case "tree_visualization":
            config_file_name = "scripts/tree_visualization/inputs/base/exp_options.yaml"
            gui_args = None
            script_type = scripts.ScriptType.TREE_VISUALIZATION
        case other:
            raise ValueError(f"Not a good name: {other}")

    # Format arguments for human-readable display
    args_display = format_gui_args_for_display(gui_args)

    chipiron_logger.info(
        "GUI Configuration Selected:\n"
        "  Script Type: %s\n"
        "  Arguments:\n    %s\n"
        "  Config File: %s",
        script_type.name if hasattr(script_type, "name") else script_type,
        args_display,
        config_file_name,
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
