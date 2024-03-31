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

    file_name_player_one: path = 'RecurZipfBase3.yaml'
    file_name_player_two: path = 'RecurZipfBase3.yaml'
    file_name_match_setting: path = 'setting_cubo.yaml'

    # For players and match modification of the yaml file specified in a respective dict
    player_one: dict[Any, Any] = field(default_factory=dict)
    player_two: dict[Any, Any] = field(default_factory=dict)
    match: dict[Any, Any] = field(default_factory=dict)
