"""CustomTkinter UI for the script launcher."""

from typing import TYPE_CHECKING, Any, cast

import customtkinter as ctk

from chipiron.environments.types import GameKind
from chipiron.players.player_ids import PlayerConfigTag

from .models import ArgsChosenByUser, ScriptGUIType
from .registries import player_options_for_game, starting_positions_for_game

if TYPE_CHECKING:
    import tkinter as tk


def _set_user_args_from_ui(
    args_chosen_by_user: ArgsChosenByUser,
    game_var: ctk.StringVar,
    chipi_algo_choice_white: ctk.StringVar,
    chipi_algo_choice_black: ctk.StringVar,
    strength_value_white: ctk.StringVar,
    strength_value_black: ctk.StringVar,
    starting_position_choice: ctk.StringVar,
) -> None:
    """Save current widget values into ArgsChosenByUser."""
    args_chosen_by_user.type = ScriptGUIType.PLAY_OR_WATCH_A_GAME
    args_chosen_by_user.game_kind = GameKind(game_var.get())

    for option in player_options_for_game(args_chosen_by_user.game_kind):
        if option.label == chipi_algo_choice_white.get():
            args_chosen_by_user.player_type_white = option.tag
            break

    for option in player_options_for_game(args_chosen_by_user.game_kind):
        if option.label == chipi_algo_choice_black.get():
            args_chosen_by_user.player_type_black = option.tag
            break

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
    args_chosen_by_user.starting_position_key = starting_position_choice.get()


def _set_tree_visualization(args_chosen_by_user: ArgsChosenByUser) -> None:
    args_chosen_by_user.type = ScriptGUIType.TREE_VISUALIZATION


def build_script_gui(root: ctk.CTk, args_chosen_by_user: ArgsChosenByUser) -> None:
    """Build the script GUI widgets and callbacks."""
    root.title("🐙 Chipiron Script Launcher 🐙")

    window_width = 900
    window_height = 400
    center_x = int(root.winfo_screenwidth() / 2 - window_width / 2)
    center_y = int(root.winfo_screenheight() / 2 - window_height / 2)
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

    cast("tk.Widget", ctk.CTkLabel(root, text="What to do?")).grid(column=0, row=0)

    cast("tk.Widget", ctk.CTkLabel(root, text="Game: ")).grid(column=0, row=1)
    cast("tk.Widget", ctk.CTkLabel(root, text="♙ White Player: ")).grid(column=0, row=2)
    cast("tk.Widget", ctk.CTkLabel(root, text="♟ Black Player: ")).grid(column=0, row=3)

    game_var = ctk.StringVar(value=args_chosen_by_user.game_kind.value)
    game_menu = ctk.CTkOptionMenu(
        master=root,
        values=[game_kind.value for game_kind in GameKind],
        variable=game_var,
    )
    cast("tk.Widget", game_menu).grid(column=1, row=1, padx=10, pady=10)

    chipi_algo_choice_white = ctk.StringVar(value="Human Player")
    chipi_algo_choice_black = ctk.StringVar(value="Human Player")
    strength_value_white = ctk.StringVar(value="1")
    strength_value_black = ctk.StringVar(value="1")
    starting_position_choice = ctk.StringVar(value="Standard")

    white_menu = ctk.CTkOptionMenu(
        master=root, values=["Human Player"], variable=chipi_algo_choice_white
    )
    cast("tk.Widget", white_menu).grid(column=3, row=2)
    black_menu = ctk.CTkOptionMenu(
        master=root, values=["Human Player"], variable=chipi_algo_choice_black
    )
    cast("tk.Widget", black_menu).grid(column=3, row=3)

    strength_options = ["1", "2", "3", "4", "5"]
    strength_label_white = ctk.CTkLabel(root, text="  strength: ")
    strength_menu_white = ctk.CTkOptionMenu(
        master=root, variable=strength_value_white, values=strength_options
    )
    cast("tk.Widget", strength_label_white).grid(column=4, row=2, padx=5, pady=10)
    cast("tk.Widget", strength_menu_white).grid(column=5, row=2, padx=10, pady=10)

    strength_label_black = ctk.CTkLabel(root, text="  strength: ")
    strength_menu_black = ctk.CTkOptionMenu(
        master=root, variable=strength_value_black, values=strength_options
    )
    cast("tk.Widget", strength_label_black).grid(column=4, row=3, padx=5, pady=10)
    cast("tk.Widget", strength_menu_black).grid(column=5, row=3, padx=10, pady=10)

    cast("tk.Widget", ctk.CTkLabel(root, text="Starting Position: ")).grid(
        column=0, row=4, padx=5, pady=10
    )
    starting_position_menu = ctk.CTkOptionMenu(
        master=root,
        values=["Standard"],
        variable=starting_position_choice,
    )
    cast("tk.Widget", starting_position_menu).grid(column=1, row=4, padx=10, pady=10)

    def option_supports_strength(label: str) -> bool:
        game_kind = GameKind(game_var.get())
        for option in player_options_for_game(game_kind):
            if option.label == label:
                return option.supports_strength
        return False

    def update_strength_visibility(*_: Any) -> None:
        if option_supports_strength(chipi_algo_choice_white.get()):
            strength_label_white.configure(text="  strength: ")
            strength_menu_white.configure(state="normal")
        else:
            strength_label_white.configure(text="")
            strength_menu_white.configure(state="disabled")

        if option_supports_strength(chipi_algo_choice_black.get()):
            strength_label_black.configure(text="  strength: ")
            strength_menu_black.configure(state="normal")
        else:
            strength_label_black.configure(text="")
            strength_menu_black.configure(state="disabled")

    def refresh_game_specific_options(*_: Any) -> None:
        game_kind = GameKind(game_var.get())
        root.title(f"🐙 Chipiron Script Launcher — {game_kind.value.capitalize()} 🐙")

        player_labels = [option.label for option in player_options_for_game(game_kind)]
        white_menu.configure(values=player_labels)
        black_menu.configure(values=player_labels)

        if "Human Player" in player_labels:
            chipi_algo_choice_white.set("Human Player")
            chipi_algo_choice_black.set("Human Player")
        else:
            chipi_algo_choice_white.set(player_labels[0])
            chipi_algo_choice_black.set(player_labels[0])

        starting_positions = starting_positions_for_game(game_kind)
        starting_labels = list(starting_positions.keys())
        starting_position_menu.configure(values=starting_labels)
        if "Standard" in starting_positions:
            starting_position_choice.set("Standard")
        else:
            starting_position_choice.set(starting_labels[0])

        update_strength_visibility()

    chipi_algo_choice_white.trace_add("write", update_strength_visibility)
    chipi_algo_choice_black.trace_add("write", update_strength_visibility)
    game_var.trace_add("write", refresh_game_specific_options)

    def on_play_or_watch() -> None:
        _set_user_args_from_ui(
            args_chosen_by_user=args_chosen_by_user,
            game_var=game_var,
            chipi_algo_choice_white=chipi_algo_choice_white,
            chipi_algo_choice_black=chipi_algo_choice_black,
            strength_value_white=strength_value_white,
            strength_value_black=strength_value_black,
            starting_position_choice=starting_position_choice,
        )
        root.destroy()

    def on_visualize_tree() -> None:
        _set_tree_visualization(args_chosen_by_user)
        root.destroy()

    play_or_watch_a_game_button = ctk.CTkButton(
        root,
        text="Play or Watch a game",
        command=on_play_or_watch,
    )
    visualize_a_tree_button = ctk.CTkButton(
        root,
        text="Visualize a tree",
        command=on_visualize_tree,
    )
    exit_button = ctk.CTkButton(root, text="Exit", command=root.quit)

    cast("tk.Widget", play_or_watch_a_game_button).grid(
        row=6, column=0, padx=10, pady=10
    )
    cast("tk.Widget", visualize_a_tree_button).grid(row=8, column=0, padx=10, pady=10)
    cast("tk.Widget", exit_button).grid(row=10, column=0, padx=10, pady=10)

    refresh_game_specific_options()
