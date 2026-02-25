"""Config-loading test for checkers piece-count player."""

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
    """Checkers config loads with defaults for optional evaluator fields."""
    pytest.importorskip("parsley")

    from chipiron.players.boardevaluators.evaluation_scale import EvaluationScale
    from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
        InternalTensorRepresentationType,
    )
    from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
    from chipiron.players.player_ids import PlayerConfigTag

    args = PlayerConfigTag.CHECKERS_TREE_PIECECOUNT.get_players_args()
    assert args.name == "CheckersTreePieceCount"
    assert isinstance(args.main_move_selector, TreeAndValueAppArgs)
    evaluator_args = args.main_move_selector.evaluator_args
    assert (
        evaluator_args.internal_representation_type
        is InternalTensorRepresentationType.NO
    )
    assert evaluator_args.master_board_evaluator.oracle_evaluation is False
    assert (
        evaluator_args.master_board_evaluator.evaluation_scale
        is EvaluationScale.ENTIRE_REAL_AXIS
    )
