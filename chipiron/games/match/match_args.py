from dataclasses import dataclass, field
from chipiron.utils import path
from typing import Any

@dataclass
class MatchArgs:
    """
    The input arguments needed by the one match script to run
    """
    # the seed
    seed: int = 0

    experiment_output_folder: path | None = None

    # path to files with yaml config the players and the match setting.
    config_file_name: path = 'scripts/one_match/inputs/base/exp_options.yaml'
    file_name_player_one: path = 'RecurZipfBase3.yaml'
    file_name_player_two: path = 'Sequool.yaml'
    file_name_match_setting: path = 'setting_duda.yaml'

    # For players and match modification of the yaml file specified in a respective dict
    player_one: dict[Any, Any] = field(default_factory=dict)
    player_two: dict[Any, Any] = field(default_factory=dict)
    match: dict[Any, Any] = field(default_factory=dict)

    print_svg_board_to_file: bool = False  # hardcode atm for webserver
