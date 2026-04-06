"""Test the script_gui functionality by simulating user inputs and creating scripts."""

from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.factory import create_script
from chipiron.scripts.gui_launcher import (
    ArgsChosenByUser,
    ScriptGUIType,
    generate_inputs,
)
from chipiron.scripts.gui_launcher.models import ParticipantSelection

args_chosen_by_user_list: list[ArgsChosenByUser] = [
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        participants=[
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
            ParticipantSelection(
                player_tag=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                strength=1,
            ),
        ],
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        participants=[
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
            ParticipantSelection(player_tag=PlayerConfigTag.UNIFORM, strength=1),
        ],
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        participants=[
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
            ParticipantSelection(player_tag=PlayerConfigTag.CHIPIRON, strength=2),
        ],
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        participants=[
            ParticipantSelection(player_tag=PlayerConfigTag.CHIPIRON, strength=2),
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
        ],
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        participants=[
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
        ],
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        participants=[
            ParticipantSelection(player_tag=PlayerConfigTag.CHIPIRON, strength=2),
            ParticipantSelection(player_tag=PlayerConfigTag.CHIPIRON, strength=2),
        ],
    ),
]


def tust_script_gui() -> None:
    """Test the script_gui functionality by simulating user inputs and creating scripts."""
    for args_chosen_by_user in args_chosen_by_user_list:
        script_type, gui_extra_args, config_file_name = generate_inputs(
            args_chosen_by_user=args_chosen_by_user
        )

        create_script(
            script_type=script_type,
            extra_args=gui_extra_args,
            config_file_name=config_file_name,
            should_parse_command_line_arguments=False,
        )


if __name__ == "__main__":
    tust_script_gui()
    print("all test passed!")
