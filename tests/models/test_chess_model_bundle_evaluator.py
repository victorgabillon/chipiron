"""Tests for the chess-specific bundle evaluator wrapper."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("coral")

from chipiron.environments.chess.players.evaluators.boardevaluators.neural_networks import (
    chess_model_bundle_evaluator,
)
from chipiron.models.model_bundle import ResolvedModelBundle


def test_create_chess_nn_state_eval_from_model_bundle_uses_generic_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chess wrapper should only add the chess converter before delegating."""
    bundle = ResolvedModelBundle(
        bundle_root="/tmp/model-bundle",
        weights_file_path="/tmp/model-bundle/weights.pt",
        architecture_file_path="/tmp/model-bundle/architecture.yaml",
        chipiron_nn_file_path="/tmp/model-bundle/chipiron_nn.yaml",
    )
    content_to_input = object()
    evaluator = object()
    calls: dict[str, object] = {}

    def fake_create_chess_content_to_input_from_bundle(
        bundle_arg: ResolvedModelBundle,
    ) -> object:
        assert bundle_arg == bundle
        return content_to_input

    def fake_create_nn_state_eval_from_model_bundle_and_converter(
        bundle_arg: ResolvedModelBundle,
        content_to_input_convert: object,
    ) -> object:
        calls["bundle_arg"] = bundle_arg
        calls["content_to_input_convert"] = content_to_input_convert
        return evaluator

    monkeypatch.setattr(
        chess_model_bundle_evaluator,
        "create_content_to_input_from_bundle",
        fake_create_chess_content_to_input_from_bundle,
    )
    monkeypatch.setattr(
        chess_model_bundle_evaluator,
        "create_nn_state_eval_from_model_bundle_and_converter",
        fake_create_nn_state_eval_from_model_bundle_and_converter,
    )

    built = chess_model_bundle_evaluator.create_chess_nn_state_eval_from_model_bundle(
        bundle
    )

    assert built is evaluator
    assert calls == {
        "bundle_arg": bundle,
        "content_to_input_convert": content_to_input,
    }
