from dataclasses import dataclass

from . import move_selector


@dataclass
class PlayerArgs:
    name: str
    main_move_selector: move_selector.AllMoveSelectorArgs
    # whether to play with syzygy when possible
    syzygy_play: bool
