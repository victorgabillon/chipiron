"""Tests for legacy NN config normalization into model bundles."""

from dataclasses import dataclass

import pytest

pytest.importorskip("PySide6")

from chipiron.environments.chess.players.evaluators.boardevaluators.neural_networks.model_bundle_normalization import (
    LegacyModelBundleNormalizationError,
    model_bundle_ref_from_legacy_nn_config,
    model_bundle_ref_from_model_weights_path,
)


@dataclass(frozen=True, slots=True)
class _LegacyNNConfig:
    """Minimal legacy config shape for normalization tests."""

    model_weights_file_name: str
    nn_architecture_args_path_to_yaml_file: str = "package://ignored/architecture.yaml"


def test_model_bundle_ref_from_legacy_hf_config() -> None:
    """HF legacy config should normalize to a folder-level bundle URI."""
    ref = model_bundle_ref_from_legacy_nn_config(
        _LegacyNNConfig(
            model_weights_file_name=(
                "hf://VictorGabillon/chipiron/prelu_no_bug/"
                "param_multi_layer_perceptron.pt@main"
            )
        )
    )

    assert ref.uri == "hf://VictorGabillon/chipiron/prelu_no_bug@main"
    assert ref.weights_file == "param_multi_layer_perceptron.pt"


def test_model_bundle_ref_from_package_weights_path() -> None:
    """Package legacy weights should normalize to a package bundle root."""
    ref = model_bundle_ref_from_model_weights_path(
        "package://data/players/board_evaluators/nn_pytorch/prelu_no_bug/weights.pt"
    )

    assert ref.uri == "package://data/players/board_evaluators/nn_pytorch/prelu_no_bug"
    assert ref.weights_file == "weights.pt"


def test_model_bundle_ref_from_local_weights_path() -> None:
    """Local legacy weights should normalize to the parent folder bundle root."""
    ref = model_bundle_ref_from_model_weights_path(
        "/tmp/models/prelu_no_bug/weights.pt"
    )

    assert ref.uri == "/tmp/models/prelu_no_bug"
    assert ref.weights_file == "weights.pt"


def test_model_bundle_ref_rejects_directory_like_weights_path() -> None:
    """Directory-like legacy weights paths should fail clearly."""
    with pytest.raises(LegacyModelBundleNormalizationError, match="must point to a file"):
        model_bundle_ref_from_model_weights_path(
            "package://data/players/board_evaluators/nn_pytorch/prelu_no_bug/"
        )
