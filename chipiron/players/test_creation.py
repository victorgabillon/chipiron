"""
Test the creation of a Chipiron player.
"""

import random
from typing import TYPE_CHECKING

from chipiron.players.factory import create_chipiron_player, create_tag_player
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.players.player_ids import PlayerConfigTag

if TYPE_CHECKING:
    from atomheart.board.utils import FenPlusHistory

    from chipiron.environments.chess.types import ChessState
    from chipiron.players.player import Player


def test_create_chipiron_player() -> None:
    """Test the creation of a Chipiron player."""

    # Create a Chipiron player with default arguments
    player: Player[FenPlusHistory, ChessState] = create_chipiron_player(
        implementation_args=ImplementationArgs(use_rust_boards=False),
        universal_behavior=True,
        random_generator=random.Random(0),
    )

    # Check if the player is created successfully
    assert player is not None
    assert player.adapter is not None

def test_tag_player_creation() -> None:
    """Test the creation of players from all PlayerConfigTag values."""

    for tag in PlayerConfigTag:
        print(f"Creating player for tag: {tag}")
        if tag == PlayerConfigTag.GUI_HUMAN:
            # GUI_HUMAN player have a specific hack logic atm
            continue

        player = create_tag_player(
            tag=tag,
            implementation_args=ImplementationArgs(use_rust_boards=False),
            universal_behavior=True,
            random_generator=random.Random(0),
        )
        assert player is not None
        assert player.adapter is not None

if __name__ == "__main__":
    test_create_chipiron_player()
    test_tag_player_creation()
    print("Player creation test passed.")
