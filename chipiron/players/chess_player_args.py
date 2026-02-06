"""Chess-specific player argument aliases."""


from chipiron.players.move_selector.chess_tree_and_value_args import (
    TreeAndValueChipironArgs,
)
from chipiron.players.move_selector.move_selector_args import NonTreeMoveSelectorArgs
from chipiron.players.player_args import PlayerArgs, PlayerFactoryArgs

type ChessMoveSelectorArgs = TreeAndValueChipironArgs | NonTreeMoveSelectorArgs
type ChessPlayerArgs = PlayerArgs
type ChessPlayerFactoryArgs = PlayerFactoryArgs
