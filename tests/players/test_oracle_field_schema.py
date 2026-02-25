import sys

import pytest


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Project runtime/parsing checks require Python >= 3.12.",
)
def test_chess_player_config_uses_oracle_fields() -> None:
    pytest.importorskip("parsley")

    from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
    from chipiron.players.player_ids import PlayerConfigTag

    args = PlayerConfigTag.RECUR_ZIPF_BASE_3.get_players_args()
    assert args.oracle_play is True

    selector = args.main_move_selector
    assert isinstance(selector, TreeAndValueAppArgs)
    assert selector.evaluator_args.master_board_evaluator.oracle_evaluation is True
