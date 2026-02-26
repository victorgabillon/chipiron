"""Test the script_gui functionality by simulating user inputs and creating scripts."""

from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.factory import create_script
from chipiron.scripts.gui_launcher import (
    ArgsChosenByUser,
    ScriptGUIType,
    generate_inputs,
)

args_chosen_by_user_list: list[ArgsChosenByUser] = [
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        player_type_white=PlayerConfigTag.GUI_HUMAN,
        player_type_black=PlayerConfigTag.RECUR_ZIPF_BASE_3,
        strength_black=1,
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        player_type_white=PlayerConfigTag.GUI_HUMAN,
        player_type_black=PlayerConfigTag.UNIFORM,
        strength_black=1,
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        player_type_white=PlayerConfigTag.GUI_HUMAN,
        player_type_black=PlayerConfigTag.CHIPIRON,
        strength_black=2,
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        player_type_black=PlayerConfigTag.GUI_HUMAN,
        player_type_white=PlayerConfigTag.CHIPIRON,
        strength_white=2,
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        player_type_black=PlayerConfigTag.GUI_HUMAN,
        player_type_white=PlayerConfigTag.GUI_HUMAN,
    ),
    ArgsChosenByUser(
        type=ScriptGUIType.PLAY_OR_WATCH_A_GAME,
        player_type_black=PlayerConfigTag.CHIPIRON,
        player_type_white=PlayerConfigTag.CHIPIRON,
        strength_white=2,
        strength_black=2,
    ),
]


def tust_script_gui() -> None:
    """Test the script_gui functionality by simulating user inputs and creating scripts."""
    for args_chosen_by_user in args_chosen_by_user_list:
        script_type, gui_extra_args, config_file_name = generate_inputs(
            args_chosen_by_user=args_chosen_by_user
        )

        # creating the script object from its name and arguments
        create_script(
            script_type=script_type,
            extra_args=gui_extra_args,
            config_file_name=config_file_name,
            should_parse_command_line_arguments=False,
        )


if __name__ == "__main__":
    tust_script_gui()
    print("all test passed!")
