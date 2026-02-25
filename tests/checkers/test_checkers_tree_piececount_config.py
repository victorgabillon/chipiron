import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Config loading in this repository targets Python >= 3.12.",
)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Config loading in this repository targets Python >= 3.12.",
)
def test_checkers_tree_piececount_yaml_loads() -> None:
    pytest.importorskip("parsley")

    from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
    from chipiron.players.player_ids import PlayerConfigTag

    args = PlayerConfigTag.CHECKERS_TREE_PIECECOUNT.get_players_args()
    assert args.name == "CheckersTreePieceCount"
    assert isinstance(args.main_move_selector, TreeAndValueAppArgs)
