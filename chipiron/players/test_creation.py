"""
Test the creation of a Chipiron player.
"""

import random

from chipiron.players.factory import create_chipiron_player
from chipiron.scripts.chipiron_args import ImplementationArgs


def test_create_chipiron_player() -> None:
    """Test the creation of a Chipiron player."""

    # Create a Chipiron player with default arguments
    player = create_chipiron_player(
        implementation_args=ImplementationArgs(use_rust_boards=False),
        universal_behavior=True,
        random_generator=random.Random(0),
    )

    # Check if the player is created successfully
    assert player is not None
    assert player.main_move_selector is not None


if __name__ == "__main__":
    test_create_chipiron_player()
