import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Checkers runtime imports require Python >= 3.12 in this repository.",
)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Checkers runtime imports require Python >= 3.12 in this repository.",
)
def test_checkers_tree_piececount_player_produces_parseable_legal_action_name() -> None:
    pytest.importorskip("atomheart")
    pytest.importorskip("valanga")
    pytest.importorskip("anemone")

    from valanga import Color
    from valanga.game import Seed

    from chipiron.environments.checkers.types import (
        CheckersDynamics,
        CheckersRules,
        CheckersState,
    )
    from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
    from chipiron.players.player_args import PlayerFactoryArgs
    from chipiron.players.player_ids import PlayerConfigTag
    from chipiron.environments.checkers.players.wiring.checkers_wiring import (
        BuildCheckersGamePlayerArgs,
        build_checkers_game_player,
    )

    player_args = PlayerConfigTag.CHECKERS_TREE_PIECECOUNT.get_players_args()
    assert isinstance(player_args.main_move_selector, TreeAndValueAppArgs)
    factory_args = PlayerFactoryArgs(player_args=player_args, seed=0)

    game_player = build_checkers_game_player(
        BuildCheckersGamePlayerArgs(
            player_factory_args=factory_args,
            player_color=Color.WHITE,
            implementation_args=object(),
            universal_behavior=True,
        )
    )

    state = CheckersState.standard()
    snapshot = state.to_text()

    recommendation = game_player.select_move_from_snapshot(
        snapshot=snapshot,
        seed=0,
        notify_percent_function=lambda _progress: None,
    )

    assert isinstance(recommendation.recommended_name, str)
    assert recommendation.recommended_name

    dynamics = CheckersDynamics(CheckersRules())
    legal_actions = dynamics.legal_actions(state).get_all()
    assert legal_actions

    parsed_action = dynamics.action_from_name(state, recommendation.recommended_name)
    assert parsed_action is not None
    parsed_name = dynamics.action_name(state, parsed_action)
    assert parsed_name == recommendation.recommended_name
