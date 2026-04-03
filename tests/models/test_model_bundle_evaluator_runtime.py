"""Tests for bundle-based chess NN evaluator runtime helpers."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("coral")

from chipiron.environments.chess.players.evaluators.boardevaluators.neural_networks import (
    chipiron_nn_args,
    model_bundle_evaluator,
)
from chipiron.models.model_bundle import ResolvedModelBundle


def test_create_nn_state_eval_from_model_bundle_uses_local_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundle evaluator should pass concrete local paths into the coral factory."""
    bundle = ResolvedModelBundle(
        bundle_root="/tmp/model-bundle",
        weights_file_path="/tmp/model-bundle/weights.pt",
        architecture_file_path="/tmp/model-bundle/architecture.yaml",
        chipiron_nn_file_path="/tmp/model-bundle/chipiron_nn.yaml",
    )
    architecture_args = object()
    content_to_input = object()
    evaluator = object()
    calls: dict[str, object] = {}

    def fake_load_nn_architecture_args_from_bundle(
        bundle_arg: ResolvedModelBundle,
    ) -> object:
        assert bundle_arg == bundle
        return architecture_args

    def fake_create_content_to_input_from_bundle(
        bundle_arg: ResolvedModelBundle,
    ) -> object:
        assert bundle_arg == bundle
        return content_to_input

    monkeypatch.setattr(
        model_bundle_evaluator,
        "load_nn_architecture_args_from_bundle",
        fake_load_nn_architecture_args_from_bundle,
    )
    monkeypatch.setattr(
        model_bundle_evaluator,
        "create_content_to_input_from_bundle",
        fake_create_content_to_input_from_bundle,
    )

    def fake_create_nn_state_eval_from_existing_model(
        *,
        model_weights_file_name: str,
        nn_architecture_args: object,
        content_to_input_convert: object,
    ) -> object:
        calls["model_weights_file_name"] = model_weights_file_name
        calls["nn_architecture_args"] = nn_architecture_args
        calls["content_to_input_convert"] = content_to_input_convert
        return evaluator

    monkeypatch.setattr(
        model_bundle_evaluator,
        "_create_nn_state_eval_from_existing_model",
        fake_create_nn_state_eval_from_existing_model,
    )

    built = model_bundle_evaluator.create_nn_state_eval_from_model_bundle(bundle)

    assert built is evaluator
    assert calls == {
        "model_weights_file_name": "/tmp/model-bundle/weights.pt",
        "nn_architecture_args": architecture_args,
        "content_to_input_convert": content_to_input,
    }


def test_create_content_to_input_from_model_weights_uses_bundle_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy weights helper should normalize and resolve bundles before loading sidecars."""
    bundle = ResolvedModelBundle(
        bundle_root="/tmp/model-bundle",
        weights_file_path="/tmp/model-bundle/weights.pt",
        architecture_file_path="/tmp/model-bundle/architecture.yaml",
        chipiron_nn_file_path="/tmp/model-bundle/chipiron_nn.yaml",
    )
    content_to_input = object()
    calls: dict[str, object] = {}

    def fake_resolve_model_bundle_from_model_weights_path(weights_path: str) -> ResolvedModelBundle:
        calls["weights_path"] = weights_path
        return bundle

    def fake_create_content_to_input_from_bundle(
        bundle_arg: ResolvedModelBundle,
    ) -> object:
        assert bundle_arg == bundle
        return content_to_input

    monkeypatch.setattr(
        chipiron_nn_args,
        "resolve_model_bundle_from_model_weights_path",
        fake_resolve_model_bundle_from_model_weights_path,
    )
    monkeypatch.setattr(
        chipiron_nn_args,
        "create_content_to_input_from_bundle",
        fake_create_content_to_input_from_bundle,
    )

    built = chipiron_nn_args.create_content_to_input_from_model_weights(
        "hf://VictorGabillon/chipiron/prelu_no_bug/weights.pt@main"
    )

    assert built is content_to_input
    assert calls == {
        "weights_path": "hf://VictorGabillon/chipiron/prelu_no_bug/weights.pt@main"
    }
