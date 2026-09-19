"""Tests for bundle-first neural-network board evaluator config parsing."""

from dataclasses import dataclass
from importlib.resources import files

import pytest

from chipiron.models.model_bundle import ModelBundleRef
from chipiron.players.boardevaluators.all_board_evaluator_args import (
    AllBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetBoardEvalArgs,
    NeuralNetModelBundleArgs,
)


@dataclass(frozen=True, slots=True)
class _BoardEvalArgsWrapper:
    """YAML wrapper for board evaluator parsing tests."""

    board_evaluator: AllBoardEvaluatorArgs


def test_direct_board_eval_config_uses_bundle_ref_without_parser() -> None:
    """Direct dataclass construction should not require the parser stack."""
    board_evaluator = NeuralNetBoardEvalArgs(
        neural_nets_model_and_architecture=NeuralNetModelBundleArgs(
            model_bundle=ModelBundleRef(
                uri="hf://VictorGabillon/chipiron/prelu_no_bug@main",
                weights_file=(
                    "param_multi_layer_perceptron_772_20_1_"
                    "parametric_relu_hyperbolic_tangent_player_to_move.pt"
                ),
            )
        )
    )

    assert board_evaluator.neural_nets_model_and_architecture.model_bundle == (
        ModelBundleRef(
            uri="hf://VictorGabillon/chipiron/prelu_no_bug@main",
            weights_file=(
                "param_multi_layer_perceptron_772_20_1_"
                "parametric_relu_hyperbolic_tangent_player_to_move.pt"
            ),
        )
    )


def test_base_board_eval_yaml_config_parses_bundle_ref() -> None:
    """The default chess NN config should expose a ModelBundleRef directly."""
    parsley = pytest.importorskip("parsley")
    package_root = files("chipiron")
    yaml_path = str(
        package_root.joinpath(
            "data/players/board_evaluator_config/base_chipiron_board_eval.yaml"
        )
    )

    wrapper: _BoardEvalArgsWrapper = parsley.resolve_yaml_file_to_base_dataclass(
        yaml_path=yaml_path,
        base_cls=_BoardEvalArgsWrapper,
        package_name=str(package_root),
    )

    assert isinstance(wrapper.board_evaluator, NeuralNetBoardEvalArgs)
    assert (
        wrapper.board_evaluator.neural_nets_model_and_architecture.model_bundle
        == ModelBundleRef(
            uri="hf://VictorGabillon/chipiron/prelu_no_bug@main",
            weights_file=(
                "param_multi_layer_perceptron_772_20_1_"
                "parametric_relu_hyperbolic_tangent_player_to_move.pt"
            ),
        )
    )
