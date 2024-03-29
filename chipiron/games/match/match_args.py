from dataclasses import dataclass, field
from typing import Any

from chipiron.utils import path


@dataclass
class MatchArgs:
    """
    The input arguments needed by the one match script to run
    """
    # the seed
    seed: int = 0

    experiment_output_folder: path | None = None

    # path to files with yaml config the players and the match setting.
    config_file_name: path = 'chipiron/scripts/one_match/inputs/base/exp_options.yaml'
    # FIXME does the lines below always overwrites the configfile name above ? is the aconfig file name ever used atm?
    file_name_player_one: path = 'RecurZipfBase3.yaml'
    file_name_player_two: path = 'Command_Line_Human.yaml'
    file_name_match_setting: path = 'setting_cubo.yaml'

    # For players and match modification of the yaml file specified in a respective dict
    player_one: dict[Any, Any] = field(default_factory=dict)
    player_two: dict[Any, Any] = field(default_factory=dict)
    match: dict[Any, Any] = field(default_factory=dict)
