from typing import Any

from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.factory import create_script
from chipiron.scripts.script_gui.script_gui_custom import generate_inputs

output_list: list[dict[str, Any]] = [
    {
        "type": "play_against_chipiron",
        "strength": 1,
        "color_human": "White",
        "chipi_algo": PlayerConfigTag.UNIFORM,
    },
    {
        "type": "play_against_chipiron",
        "strength": 2,
        "color_human": "Black",
        "chipi_algo": PlayerConfigTag.CHIPIRON,
    },
    {
        "type": "play_against_chipiron",
        "strength": 2,
        "color_human": "Black",
        "chipi_algo": PlayerConfigTag.SEQUOOL,
    },
    {"type": "play_two_humans"},
    {"type": "watch_a_game"},
]


def test_script_gui() -> None:
    for output in output_list:
        script_type, gui_extra_args, config_file_name = generate_inputs(output=output)

        # creating the script object from its name and arguments
        create_script(
            script_type=script_type,
            extra_args=gui_extra_args,
            config_file_name=config_file_name,
            should_parse_command_line_arguments=False,
        )


if __name__ == "__main__":
    test_script_gui()
    print("all test passed!")
