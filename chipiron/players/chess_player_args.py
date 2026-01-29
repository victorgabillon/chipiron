"""Chess-specific player argument aliases."""


from typing import TypeAlias

from chipiron.players.move_selector.chess_tree_and_value_args import (
    TreeAndValueChipironArgs,
)
from chipiron.players.move_selector.factory import NonTreeMoveSelectorArgs
from chipiron.players.player_args import PlayerArgs, PlayerFactoryArgs

ChessMoveSelectorArgs: TypeAlias = TreeAndValueChipironArgs | NonTreeMoveSelectorArgs
ChessPlayerArgs: TypeAlias = PlayerArgs
ChessPlayerFactoryArgs: TypeAlias = PlayerFactoryArgs
