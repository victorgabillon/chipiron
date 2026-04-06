"""CustomTkinter UI for the script launcher."""

from dataclasses import dataclass
from typing import Any, cast

import customtkinter as _customtkinter  # type: ignore[reportMissingImports]

from chipiron.environments.types import GameKind

from .logic import apply_game_kind_defaults
from .models import ArgsChosenByUser, ParticipantSelection, ScriptGUIType
from .registries import (
    launcher_spec_for_game,
    player_label_for_tag,
    player_option_for_label,
)

ctk: Any = _customtkinter
STRENGTH_OPTIONS: tuple[str, ...] = ("1", "2", "3", "4", "5")


@dataclass(frozen=True, slots=True)
class ParticipantRowModel:
    """Pure UI model for one participant row."""

    label_text: str
    player_labels: tuple[str, ...]
    selected_player_label: str
    show_strength: bool
    strength_value: str


@dataclass(slots=True)
class ParticipantRowWidgets:
    """Widget bundle for one participant row."""

    label_widget: Any
    player_var: Any
    player_menu: Any
    strength_label: Any
    strength_var: Any
    strength_menu: Any

    def destroy(self) -> None:
        """Destroy the widgets owned by this row."""
        self.label_widget.destroy()
        self.player_menu.destroy()
        self.strength_label.destroy()
        self.strength_menu.destroy()


def participant_row_models_for_state(
    args_chosen_by_user: ArgsChosenByUser,
) -> tuple[ParticipantRowModel, ...]:
    """Build pure row models from the launcher state."""
    launcher_spec = launcher_spec_for_game(args_chosen_by_user.game_kind)
    player_labels = tuple(option.label for option in launcher_spec.player_options)
    row_models: list[ParticipantRowModel] = []

    for index, participant in enumerate(args_chosen_by_user.participants):
        selected_player_label = player_label_for_tag(
            args_chosen_by_user.game_kind,
            participant.player_tag,
        )
        row_models.append(
            ParticipantRowModel(
                label_text=(
                    launcher_spec.participant_labels[index]
                    if index < len(launcher_spec.participant_labels)
                    else f"Participant {index + 1}"
                ),
                player_labels=player_labels,
                selected_player_label=selected_player_label,
                show_strength=player_option_for_label(
                    args_chosen_by_user.game_kind,
                    selected_player_label,
                ).supports_strength,
                strength_value=str(
                    participant.strength if participant.strength is not None else 1
                ),
            )
        )

    return tuple(row_models)


def _set_user_args_from_ui(
    args_chosen_by_user: ArgsChosenByUser,
    game_var: Any,
    participant_rows: list[ParticipantRowWidgets],
    starting_position_choice: Any,
) -> None:
    """Save current widget values into ArgsChosenByUser."""
    game_kind = GameKind(cast(str, game_var.get()))
    args_chosen_by_user.type = ScriptGUIType.PLAY_OR_WATCH_A_GAME
    args_chosen_by_user.game_kind = game_kind
    participants: list[ParticipantSelection] = []
    for row in participant_rows:
        option = player_option_for_label(game_kind, cast(str, row.player_var.get()))
        participants.append(
            ParticipantSelection(
                player_tag=option.tag,
                strength=int(row.strength_var.get()) if option.supports_strength else None,
            )
        )
    args_chosen_by_user.participants = participants
    args_chosen_by_user.starting_position_key = cast(str, starting_position_choice.get())


def _set_tree_visualization(args_chosen_by_user: ArgsChosenByUser) -> None:
    """Switch launcher state to tree-visualization mode."""
    args_chosen_by_user.type = ScriptGUIType.TREE_VISUALIZATION


def build_script_gui(root: Any, args_chosen_by_user: ArgsChosenByUser) -> None:
    """Build the script GUI widgets and callbacks."""
    root.title("🐙 Chipiron Script Launcher 🐙")

    window_width = 900
    window_height = 400
    center_x = int(root.winfo_screenwidth() / 2 - window_width / 2)
    center_y = int(root.winfo_screenheight() / 2 - window_height / 2)
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

    ctk.CTkLabel(root, text="What to do?").grid(column=0, row=0)

    ctk.CTkLabel(root, text="Game: ").grid(column=0, row=1, padx=10, pady=10)

    game_var = ctk.StringVar(value=args_chosen_by_user.game_kind.value)
    game_menu = ctk.CTkOptionMenu(
        master=root,
        values=[game_kind.value for game_kind in GameKind],
        variable=game_var,
    )
    game_menu.grid(column=1, row=1, padx=10, pady=10)

    participants_frame = ctk.CTkFrame(root, fg_color="transparent")
    participants_frame.grid(column=0, row=2, columnspan=6, padx=10, pady=10, sticky="w")

    ctk.CTkLabel(root, text="Starting Position: ").grid(
        column=0, row=3, padx=5, pady=10
    )
    starting_position_choice = ctk.StringVar(value=args_chosen_by_user.starting_position_key)
    starting_position_menu = ctk.CTkOptionMenu(
        master=root,
        values=[args_chosen_by_user.starting_position_key],
        variable=starting_position_choice,
    )
    starting_position_menu.grid(column=1, row=3, padx=10, pady=10)

    participant_rows: list[ParticipantRowWidgets] = []

    def update_strength_visibility(row_widgets: ParticipantRowWidgets) -> None:
        option = player_option_for_label(
            GameKind(cast(str, game_var.get())),
            cast(str, row_widgets.player_var.get()),
        )
        if option.supports_strength:
            if not row_widgets.strength_var.get():
                row_widgets.strength_var.set(STRENGTH_OPTIONS[0])
            row_widgets.strength_label.grid()
            row_widgets.strength_menu.grid()
            row_widgets.strength_menu.configure(state="normal")
        else:
            row_widgets.strength_label.grid_remove()
            row_widgets.strength_menu.grid_remove()
            row_widgets.strength_menu.configure(state="disabled")

    def rebuild_participant_rows() -> None:
        participant_row_models = participant_row_models_for_state(args_chosen_by_user)

        for row_widgets in participant_rows:
            row_widgets.destroy()
        participant_rows.clear()

        for row_index, row_model in enumerate(participant_row_models):
            label_widget = ctk.CTkLabel(
                participants_frame,
                text=f"{row_model.label_text} Participant: ",
            )
            label_widget.grid(column=0, row=row_index, padx=5, pady=10, sticky="w")

            player_var = ctk.StringVar(value=row_model.selected_player_label)
            player_menu = ctk.CTkOptionMenu(
                master=participants_frame,
                values=list(row_model.player_labels),
                variable=player_var,
            )
            player_menu.grid(column=1, row=row_index, padx=10, pady=10)

            strength_label = ctk.CTkLabel(participants_frame, text="strength:")
            strength_label.grid(column=2, row=row_index, padx=5, pady=10)

            strength_var = ctk.StringVar(value=row_model.strength_value)
            strength_menu = ctk.CTkOptionMenu(
                master=participants_frame,
                variable=strength_var,
                values=list(STRENGTH_OPTIONS),
            )
            strength_menu.grid(column=3, row=row_index, padx=10, pady=10)

            row_widgets = ParticipantRowWidgets(
                label_widget=label_widget,
                player_var=player_var,
                player_menu=player_menu,
                strength_label=strength_label,
                strength_var=strength_var,
                strength_menu=strength_menu,
            )
            participant_rows.append(row_widgets)

            def on_player_choice_changed(
                *_ignored: object,
                row_widgets: ParticipantRowWidgets = row_widgets,
            ) -> None:
                update_strength_visibility(row_widgets)

            player_var.trace_add("write", on_player_choice_changed)
            update_strength_visibility(row_widgets)
            if not row_model.show_strength:
                row_widgets.strength_label.grid_remove()
                row_widgets.strength_menu.grid_remove()

    def refresh_game_specific_options() -> None:
        game_kind = GameKind(cast(str, game_var.get()))
        args_chosen_by_user.game_kind = game_kind

        launcher_spec = launcher_spec_for_game(game_kind)
        root.title(f"🐙 Chipiron Script Launcher — {game_kind.value.capitalize()} 🐙")

        rebuild_participant_rows()

        starting_labels = list(launcher_spec.starting_positions.keys())
        starting_position_menu.configure(values=starting_labels)
        if args_chosen_by_user.starting_position_key in launcher_spec.starting_positions:
            starting_position_choice.set(args_chosen_by_user.starting_position_key)
        else:
            starting_position_choice.set(launcher_spec.default_starting_position_key)

    def on_game_kind_changed(*_args: Any) -> None:
        args_chosen_by_user.game_kind = GameKind(cast(str, game_var.get()))
        apply_game_kind_defaults(args_chosen_by_user)
        refresh_game_specific_options()

    game_var.trace_add("write", on_game_kind_changed)

    def on_play_or_watch() -> None:
        _set_user_args_from_ui(
            args_chosen_by_user=args_chosen_by_user,
            game_var=game_var,
            participant_rows=participant_rows,
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

    play_or_watch_a_game_button.grid(row=5, column=0, padx=10, pady=10)
    visualize_a_tree_button.grid(row=6, column=0, padx=10, pady=10)
    exit_button.grid(row=7, column=0, padx=10, pady=10)

    refresh_game_specific_options()
