"""Describe script defines a graphical user interface (GUI) for interacting with the chipiron program.

The GUI allows the user to select various options and perform different actions such as playing against the chipiron AI, watching a game, or visualizing a tree.

The script_gui function creates the GUI window and handles the user interactions.
The play_against_chipiron, watch_a_game, and visualize_a_tree functions are called when the corresponding buttons are clicked.

Note: This script requires the customtkinter and chipiron modules to be installed.
"""

# TODO switch to pygame
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import customtkinter as ctk
from anemone import TreeAndValuePlayerArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from parsley_coco import make_partial_dataclass_with_optional_paths

from chipiron import scripts
from chipiron.environments.chess.starting_position_args import (
    FenStartingPositionArgs,
    StartingPositionArgsType,
)
from chipiron.games.game.game_args import GameArgs
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_settings_args import MatchSettingsArgs
from chipiron.games.match.match_tag import MatchConfigTag
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.one_match.one_match import MatchScriptArgs
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    import tkinter as tk


# Move this to module level for access by callback functions
CHIPI_ALGO_OPTIONS_LIST: list[tuple[str, PlayerConfigTag]] = [
    ("RecurZipfBase3", PlayerConfigTag.RECUR_ZIPF_BASE_3),
    ("Uniform", PlayerConfigTag.UNIFORM),
    ("Sequool", PlayerConfigTag.SEQUOOL),
    ("Human Player", PlayerConfigTag.GUI_HUMAN),
]


def format_gui_args_for_display(gui_args: Any) -> str:
    """Format GUI arguments for human-readable display in logging.

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
                    if hasattr(criterion, "tree_branch_limit"):
                        args_parts.append(
                            f"Player One Strength: {criterion.tree_branch_limit} moves"
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
                    if hasattr(criterion, "tree_branch_limit"):
                        args_parts.append(
                            f"Player Two Strength: {criterion.tree_branch_limit} moves"
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


class MatchGUIStartingPosition(StrEnum):
    """The starting position for the chess match."""

    STANDARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ENDGAME = "6k1/p7/8/8/7N/7K/2N5/8 w - - 0 1"

    def get_fen(self) -> str:
        """Get the FEN string corresponding to the starting position.

        Returns:
            str: The FEN string for the starting position.

        """
        return str(self.value)


STARTING_POSITION_OPTIONS_DICT: dict[str, MatchGUIStartingPosition] = {
    "Standard": MatchGUIStartingPosition.STANDARD,
    "End game": MatchGUIStartingPosition.ENDGAME,
}


class ScriptGUIType(StrEnum):
    """The type of script to run based on GUI selection."""

    PLAY_OR_WATCH_A_GAME = "play_or_watch_a_game"
    TREE_VISUALIZATION = "tree_visualization"


@dataclass
class ArgsChosenByUser:
    """The arguments chosen by the user in the GUI."""

    type: ScriptGUIType = ScriptGUIType.PLAY_OR_WATCH_A_GAME
    player_type_white: PlayerConfigTag = PlayerConfigTag.RECUR_ZIPF_BASE_3
    strength_white: int | None = 1
    player_type_black: PlayerConfigTag = PlayerConfigTag.RECUR_ZIPF_BASE_3
    strength_black: int | None = 1
    starting_position: MatchGUIStartingPosition = (
        MatchGUIStartingPosition.STANDARD
    )  # "Standard" or "Endgame"


def script_gui() -> tuple[scripts.ScriptType, IsDataclass | None, str]:
    """Create a graphical user interface (GUI) for interacting with the chipiron program.

    Returns:
        A tuple containing the script type and the arguments for the selected action.

    """
    root: ctk.CTk = ctk.CTk()
    args_chosen_by_user: ArgsChosenByUser = ArgsChosenByUser()
    # place a label on the root window
    root.title("🐙 chess with chipirons 🐙")

    # frm = ctk.CTkFrame(root, padding=10)

    window_width: int = 900
    window_height: int = 400

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

    message = cast("tk.Widget", ctk.CTkLabel(root, text="♙ White Player: "))
    message.grid(column=0, row=2)

    message = cast("tk.Widget", ctk.CTkLabel(root, text="♟ Black Player: "))
    message.grid(column=0, row=3)

    # Create the list of options
    chipi_algo_options_list: list[tuple[str, PlayerConfigTag]] = CHIPI_ALGO_OPTIONS_LIST

    ######PLAYER WHITE

    # Variable to keep track of the option
    # selected in OptionMenu
    chipi_algo_choice_white = ctk.StringVar(
        value=args_chosen_by_user.player_type_white
    )  # Use display name as default

    # Helper to get display names and values
    algo_display_names = [name for name, _ in chipi_algo_options_list]

    # Create the option menu widget and passing the display names
    algo_menu = ctk.CTkOptionMenu(
        master=root, values=algo_display_names, variable=chipi_algo_choice_white
    )
    cast("tk.Widget", algo_menu).grid(column=3, row=2)

    strength_label = ctk.CTkLabel(root, text="  strength: ")

    # Create the list of options
    options_list = ["1", "2", "3", "4", "5"]

    # Variable to keep track of the option
    # selected in OptionMenu
    strength_value_white = ctk.StringVar(value="1")  # set initial value

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = ctk.CTkOptionMenu(
        master=root, variable=strength_value_white, values=options_list
    )

    # Always grid the strength controls to maintain layout consistency
    cast("tk.Widget", strength_label).grid(column=4, row=2, padx=5, pady=10)
    cast("tk.Widget", strength_menu).grid(column=5, row=2, padx=10, pady=10)

    # When reading the value, map display name to PlayerConfigTag
    def get_selected_algo_tag() -> PlayerConfigTag:
        selected_name = chipi_algo_choice_white.get()
        for name, tag in chipi_algo_options_list:
            if name == selected_name:
                return tag
        return PlayerConfigTag.RECUR_ZIPF_BASE_3  # fallback

    def update_strength_menu(*_: Any) -> None:
        # Only show strength menu and label if not human
        current_tag = get_selected_algo_tag()
        if current_tag != PlayerConfigTag.GUI_HUMAN:
            strength_label.configure(text="  strength: ")
            strength_menu.configure(state="normal")
            cast("tk.Widget", strength_label).grid(column=4, row=2, padx=5, pady=10)
            cast("tk.Widget", strength_menu).grid(column=5, row=2, padx=10, pady=10)
        else:
            # Make them invisible but keep the space
            strength_label.configure(text="")
            strength_menu.grid_remove()

    chipi_algo_choice_white.trace_add("write", update_strength_menu)
    update_strength_menu()

    ######PLAYER BLACK

    # Variable to keep track of the option
    # selected in OptionMenu
    chipi_algo_choice_black = ctk.StringVar(
        value=args_chosen_by_user.player_type_black
    )  # set initial value

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    algo_menu_black = ctk.CTkOptionMenu(
        master=root, values=algo_display_names, variable=chipi_algo_choice_black
    )
    cast("tk.Widget", algo_menu_black).grid(column=3, row=3)

    strength_label_black = ctk.CTkLabel(root, text="  strength: ")

    # Variable to keep track of the option
    # selected in OptionMenu
    strength_value_black = ctk.StringVar(value="1")  # set initial value

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu_black = ctk.CTkOptionMenu(
        master=root, variable=strength_value_black, values=options_list
    )

    # Always grid the strength controls to maintain layout consistency
    cast("tk.Widget", strength_label_black).grid(column=4, row=3, padx=5, pady=10)
    cast("tk.Widget", strength_menu_black).grid(column=5, row=3, padx=10, pady=10)

    # When reading the value, map display name to PlayerConfigTag for black player
    def get_selected_algo_tag_black() -> PlayerConfigTag:
        selected_name = chipi_algo_choice_black.get()
        for name, tag in chipi_algo_options_list:
            if name == selected_name:
                return tag
        return PlayerConfigTag.RECUR_ZIPF_BASE_3  # fallback

    def update_strength_menu_black(*_: Any) -> None:
        # Only show strength menu and label if not human
        current_tag_black = get_selected_algo_tag_black()
        if current_tag_black != PlayerConfigTag.GUI_HUMAN:
            # Make them visible
            strength_label_black.configure(text="  strength: ")
            strength_menu_black.configure(state="normal")
            cast("tk.Widget", strength_label_black).grid(
                column=4, row=3, padx=5, pady=10
            )
            cast("tk.Widget", strength_menu_black).grid(
                column=5, row=3, padx=10, pady=10
            )
        else:
            # Make them invisible but keep the space
            strength_label_black.configure(text="")
            strength_menu_black.grid_remove()

    chipi_algo_choice_black.trace_add("write", update_strength_menu_black)
    update_strength_menu_black()

    ######STARTING POSITION

    # Label for starting position
    starting_position_label = ctk.CTkLabel(root, text="Starting Position: ")
    cast("tk.Widget", starting_position_label).grid(column=0, row=4, padx=5, pady=10)

    # Variable to keep track of the starting position selection
    starting_position_choice = ctk.StringVar(value="Standard")

    # Starting position options
    starting_position_display_names = list(STARTING_POSITION_OPTIONS_DICT.keys())

    # Create the option menu widget for starting position
    starting_position_menu = ctk.CTkOptionMenu(
        master=root,
        values=starting_position_display_names,
        variable=starting_position_choice,
    )
    cast("tk.Widget", starting_position_menu).grid(column=1, row=4, padx=10, pady=10)

    # play or watch button
    play_or_watch_a_game_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text="Play or Watch a game",
        command=lambda: [
            play_or_watch_a_game(
                args_chosen_by_user=args_chosen_by_user,
                chipi_algo_choice_white=chipi_algo_choice_white,
                chipi_algo_choice_black=chipi_algo_choice_black,
                strength_value_white=strength_value_white,
                strength_value_black=strength_value_black,
                starting_position_choice=starting_position_choice,
            ),
            root.destroy(),
        ],
    )

    # visualize button
    visualize_a_tree_button: ctk.CTkButton = ctk.CTkButton(
        root,
        text="Visualize a tree",
        command=lambda: [visualize_a_tree(args_chosen_by_user), root.destroy()],
    )

    cast("tk.Widget", play_or_watch_a_game_button).grid(
        row=6, column=0, padx=10, pady=10
    )
    cast("tk.Widget", visualize_a_tree_button).grid(row=8, column=0, padx=10, pady=10)
    cast("tk.Widget", exit_button).grid(row=10, column=0, padx=10, pady=10)

    cast("tk.Tk", root).mainloop()

    return generate_inputs(args_chosen_by_user=args_chosen_by_user)


def generate_inputs(
    args_chosen_by_user: ArgsChosenByUser,
) -> tuple[scripts.ScriptType, IsDataclass | None, str]:
    """Generate script type and arguments based on the user input.

    Args:
        args_chosen_by_user (ArgsChosenByUser): The arguments chosen by the user.

    Returns:
        tuple[scripts.ScriptType, IsDataclass | None, str]: The script type, arguments, and config file name.

    """
    script_type: scripts.ScriptType

    partial_op_match_script_args = make_partial_dataclass_with_optional_paths(
        cls=MatchScriptArgs
    )
    partial_op_match_args = make_partial_dataclass_with_optional_paths(cls=MatchArgs)

    partial_op_match_settings_args = make_partial_dataclass_with_optional_paths(
        cls=MatchSettingsArgs
    )
    partial_op_game_args = make_partial_dataclass_with_optional_paths(cls=GameArgs)

    partial_op_player_args = make_partial_dataclass_with_optional_paths(cls=PlayerArgs)

    partial_op_base_script_args = make_partial_dataclass_with_optional_paths(
        cls=BaseScriptArgs
    )

    partial_op_tree_and_value_player_args = make_partial_dataclass_with_optional_paths(
        cls=TreeAndValuePlayerArgs
    )
    partial_op_tree_branch_limit_args = make_partial_dataclass_with_optional_paths(
        cls=TreeBranchLimitArgs
    )

    config_file_name: str
    match args_chosen_by_user.type:
        case ScriptGUIType.PLAY_OR_WATCH_A_GAME:
            gui_args = partial_op_match_script_args(
                gui=True,
                base_script_args=partial_op_base_script_args(profiling=False, seed=0),
                match_args=partial_op_match_args(
                    match_setting=MatchConfigTag.DUDA,
                    match_setting_overwrite=partial_op_match_settings_args(
                        game_args=partial_op_game_args(
                            starting_position=FenStartingPositionArgs(
                                type=StartingPositionArgsType.FEN,
                                fen=args_chosen_by_user.starting_position,
                            )
                        )
                    ),
                ),
            )
            config_file_name = "chipiron/scripts/one_match/inputs/human_play_against_computer/exp_options.yaml"

            gui_args.match_args.player_one = PlayerConfigTag(
                args_chosen_by_user.player_type_white
            )
            gui_args.match_args.player_two = PlayerConfigTag(
                args_chosen_by_user.player_type_black
            )
            if args_chosen_by_user.player_type_white != PlayerConfigTag.GUI_HUMAN:
                assert args_chosen_by_user.strength_white is not None
                tree_branch_limit_white: int = (
                    4 * 10**args_chosen_by_user.strength_white
                )
                gui_args.match_args.player_one_overwrite = partial_op_player_args(
                    main_move_selector=partial_op_tree_and_value_player_args(
                        type=MoveSelectorTypes.TreeAndValue,
                        stopping_criterion=partial_op_tree_branch_limit_args(
                            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                            tree_branch_limit=tree_branch_limit_white,
                        ),
                    )
                )
            if args_chosen_by_user.player_type_black != PlayerConfigTag.GUI_HUMAN:
                assert args_chosen_by_user.strength_black is not None
                tree_branch_limit_black: int = (
                    4 * 10**args_chosen_by_user.strength_black
                )
                gui_args.match_args.player_two_overwrite = partial_op_player_args(
                    main_move_selector=partial_op_tree_and_value_player_args(
                        type=MoveSelectorTypes.TreeAndValue,
                        stopping_criterion=partial_op_tree_branch_limit_args(
                            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                            tree_branch_limit=tree_branch_limit_black,
                        ),
                    )
                )

            script_type = scripts.ScriptType.ONE_MATCH
        case ScriptGUIType.TREE_VISUALIZATION:
            config_file_name = "scripts/tree_visualization/inputs/base/exp_options.yaml"
            gui_args = None
            script_type = scripts.ScriptType.TREE_VISUALIZATION

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


def play_or_watch_a_game(
    args_chosen_by_user: ArgsChosenByUser,
    chipi_algo_choice_white: ctk.StringVar,
    chipi_algo_choice_black: ctk.StringVar,
    strength_value_white: ctk.StringVar,
    strength_value_black: ctk.StringVar,
    starting_position_choice: ctk.StringVar,
) -> bool:
    """Handle the "Play or Watch a game" button callback.

    Sets the output dictionary with the selected options for playing or watching a game.

    Args:
        output: The output dictionary to store the selected options.

    Returns:
        True.

    """
    args_chosen_by_user.type = ScriptGUIType.PLAY_OR_WATCH_A_GAME

    # Map display names to PlayerConfigTag enums
    selected_white = chipi_algo_choice_white.get()
    for name, tag in CHIPI_ALGO_OPTIONS_LIST:
        if name == selected_white:
            args_chosen_by_user.player_type_white = tag
            break
    else:
        args_chosen_by_user.player_type_white = (
            PlayerConfigTag.RECUR_ZIPF_BASE_3
        )  # fallback

    selected_black = chipi_algo_choice_black.get()
    for name, tag in CHIPI_ALGO_OPTIONS_LIST:
        if name == selected_black:
            args_chosen_by_user.player_type_black = tag
            break
    else:
        args_chosen_by_user.player_type_black = (
            PlayerConfigTag.RECUR_ZIPF_BASE_3
        )  # fallback

    args_chosen_by_user.strength_white = (
        int(strength_value_white.get())
        if args_chosen_by_user.player_type_white != PlayerConfigTag.GUI_HUMAN
        else None
    )
    args_chosen_by_user.strength_black = (
        int(strength_value_black.get())
        if args_chosen_by_user.player_type_black != PlayerConfigTag.GUI_HUMAN
        else None
    )

    # Capture starting position choice
    args_chosen_by_user.starting_position = STARTING_POSITION_OPTIONS_DICT[
        starting_position_choice.get()
    ]

    return True


def visualize_a_tree(args_chosen_by_user: ArgsChosenByUser) -> bool:
    """Handle the "Visualize a tree" button callback.

    Sets the output dictionary with the selected options for tree visualization.

    Args:
        output: The output dictionary to store the selected options.

    Returns:
        True.

    """
    args_chosen_by_user.type = ScriptGUIType.TREE_VISUALIZATION
    return True
