"""Portable evaluator configuration, legacy representations and public scores."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import dacite
import pytest
import torch
from atomheart.games.morpion import initial_state
from valanga import SOLO
from valanga.evaluations import Certainty

from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
    load_morpion_evaluator_from_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_evaluator_args import (
    MorpionNeuralEvaluatorArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    MorpionRelationalEntityTokenConverter,
    build_morpion_regressor,
    load_morpion_model_bundle,
    morpion_evaluator_v1_model_args,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.target_transform import (
    InvalidMorpionTargetTransformError,
)
from chipiron.environments.morpion.players.evaluators.wiring import MorpionEvalWiring
from chipiron.environments.morpion.players.wiring.morpion_wiring import (
    BuildMorpionGamePlayerArgs,
    build_morpion_game_player,
)
from chipiron.environments.morpion.types import MorpionState
from chipiron.players.boardevaluators.master_board_evaluator_args import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.player_args import PlayerArgs, PlayerFactoryArgs
from tests.environments.test_morpion_integration import make_tree_and_value_selector

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("promoted", [False, True])
@pytest.mark.parametrize("standardized", [False, True])
def test_bundle_public_scores_and_terminal_override(
    tmp_path: Path, *, promoted: bool, standardized: bool
) -> None:
    """Both representations and target scales work through ordinary evaluation."""
    args = replace(
        morpion_evaluator_v1_model_args(),
        latent_window_move_features="promoted_only" if promoted else "none",
        entity_input_feature_dim=26 if promoted else 25,
        target_transform_enabled=standardized,
        target_mean=60.0 if standardized else 0.0,
        target_standard_deviation=4.0 if standardized else 1.0,
    )
    torch.manual_seed(0)
    model = build_morpion_regressor(args).eval()
    state = MorpionState.from_atomheart_state(initial_state())
    converter = MorpionRelationalEntityTokenConverter(
        latent_window_move_features=args.latent_window_move_features
    )
    inputs = converter.state_to_model_input_tensors(state)
    with torch.inference_mode():
        internal = model.forward_normalized(*inputs)
        expected = model(*inputs)
    torch.testing.assert_close(
        expected, 60.0 + 4.0 * internal if standardized else internal
    )
    bundle = tmp_path / "portable_bundle"
    save_morpion_model_bundle(model, bundle, model_args=args)
    restored, restored_args, manifest = load_morpion_model_bundle(bundle)
    restored.eval()
    assert restored_args == args
    assert manifest.input_dim == (26 if promoted else 25)
    with torch.inference_mode():
        assert torch.equal(restored(*inputs), expected)
    evaluator = MorpionEvalWiring(
        neural=MorpionNeuralEvaluatorArgs(model_bundle=str(bundle))
    ).build_chi()
    assert evaluator.evaluate(state).score == pytest.approx(float(expected.reshape(())))
    terminal = MorpionState(
        points=frozenset(),
        used_unit_segments=frozenset(),
        dir_usage_entries=(),
        moves=17,
        is_terminal=True,
    )
    value = evaluator.evaluate(terminal)
    assert value.score == 17.0 and value.certainty is Certainty.TERMINAL


def test_v1_specification_and_legacy_defaults() -> None:
    """The named preset uses scale 0.25; legacy construction still uses 1.0."""
    args = morpion_evaluator_v1_model_args()
    assert args.entity_d_model == 64 and args.entity_n_layer == 2
    assert args.entity_n_head == 4 and args.entity_dim_feedforward == 256
    assert args.relation_bias_scale == 0.25 and args.entity_pooling == "value_token"
    assert args.global_geometry_features == "none"
    assert args.edge_token_mode == "drawn_only"
    assert args.latent_window_move_features == "none"
    assert not args.target_transform_enabled
    assert sum(p.numel() for p in build_morpion_regressor(args).parameters()) == 106049
    assert MorpionRegressorArgs().relation_bias_scale == 1.0


def test_serialized_config_reaches_normal_player_wiring(tmp_path: Path) -> None:
    """The standard evaluator union decodes the neural config and the player uses it."""
    bundle = tmp_path / "bundle"
    args = morpion_evaluator_v1_model_args()
    save_morpion_model_bundle(build_morpion_regressor(args), bundle, model_args=args)
    config = MorpionNeuralEvaluatorArgs(model_bundle=str(bundle))
    data = json.loads(json.dumps({"board_evaluator": asdict(config)}))
    master = dacite.from_dict(MasterBoardEvaluatorArgs, data)
    assert master.board_evaluator == config
    selector = make_tree_and_value_selector()
    selector.evaluator_args.master_board_evaluator = master
    factory = PlayerFactoryArgs(
        player_args=PlayerArgs(
            name="MorpionEvaluatorV1", main_move_selector=selector, oracle_play=False
        ),
        seed=0,
    )
    build_args = BuildMorpionGamePlayerArgs(
        player_factory_args=factory,
        player_role=SOLO,
        implementation_args=None,
        universal_behavior=False,
    )
    assert build_morpion_game_player(build_args) is not None
    # A broken configured path must fail, proving that wiring does not silently
    # fall back to the heuristic after decoding a neural configuration.
    master.board_evaluator = replace(config, model_bundle=str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError):
        build_morpion_game_player(build_args)


def test_legacy_runtime_import_still_reexports_normal_loader() -> None:
    """Existing bootstrap callers use the same implementation as normal inference."""
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        load_morpion_evaluator_from_model_bundle as legacy_loader,
    )

    assert legacy_loader is load_morpion_evaluator_from_model_bundle


@pytest.mark.parametrize("standard_deviation", [0.0, -1.0, float("nan")])
def test_invalid_standardized_target_metadata_is_rejected(
    standard_deviation: float,
) -> None:
    """Invalid persisted scaling cannot silently distort public values."""
    with pytest.raises(InvalidMorpionTargetTransformError):
        replace(
            morpion_evaluator_v1_model_args(),
            target_transform_enabled=True,
            target_standard_deviation=standard_deviation,
        )
