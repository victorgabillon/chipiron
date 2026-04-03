"""Tests for generic bundle-based NN runtime helpers."""

from dataclasses import dataclass

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("dacite")

from chipiron.models.model_bundle import ResolvedModelBundle
from chipiron.players.boardevaluators.neural_networks import model_bundle_runtime


@dataclass(frozen=True, slots=True)
class _FakeArchitectureArgs:
    """Small dataclass used to test generic architecture loading."""

    width: int


def test_load_nn_architecture_args_from_file() -> None:
    """Architecture loading should parse YAML into the runtime dataclass."""
    calls: dict[str, object] = {}

    def fake_yaml_fetch_args_in_file(path_file: str) -> dict[str, int]:
        calls["path_file"] = path_file
        return {"width": 7}

    original_fetch = model_bundle_runtime.yaml_fetch_args_in_file
    original_class_getter = model_bundle_runtime._get_neural_net_architecture_args_class
    model_bundle_runtime.yaml_fetch_args_in_file = fake_yaml_fetch_args_in_file
    model_bundle_runtime._get_neural_net_architecture_args_class = (
        lambda: _FakeArchitectureArgs
    )
    try:
        architecture_args = model_bundle_runtime.load_nn_architecture_args_from_file(
            "/tmp/model-bundle/architecture.yaml"
        )
    finally:
        model_bundle_runtime.yaml_fetch_args_in_file = original_fetch
        model_bundle_runtime._get_neural_net_architecture_args_class = (
            original_class_getter
        )

    assert architecture_args == _FakeArchitectureArgs(width=7)
    assert calls == {"path_file": "/tmp/model-bundle/architecture.yaml"}


def test_load_nn_architecture_args_from_bundle() -> None:
    """Bundle architecture loading should delegate through the bundle path."""
    bundle = ResolvedModelBundle(
        bundle_root="/tmp/model-bundle",
        weights_file_path="/tmp/model-bundle/weights.pt",
        architecture_file_path="/tmp/model-bundle/architecture.yaml",
        chipiron_nn_file_path="/tmp/model-bundle/chipiron_nn.yaml",
    )
    original_loader = model_bundle_runtime.load_nn_architecture_args_from_file
    calls: dict[str, object] = {}

    def fake_load_nn_architecture_args_from_file(path: str) -> object:
        calls["path"] = path
        return object()

    model_bundle_runtime.load_nn_architecture_args_from_file = (
        fake_load_nn_architecture_args_from_file
    )
    try:
        _ = model_bundle_runtime.load_nn_architecture_args_from_bundle(bundle)
    finally:
        model_bundle_runtime.load_nn_architecture_args_from_file = original_loader

    assert calls == {"path": "/tmp/model-bundle/architecture.yaml"}


def test_create_nn_state_eval_from_model_bundle_and_converter_uses_local_paths() -> None:
    """Generic runtime should pass resolved local paths and converter to coral."""
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

    original_bundle_loader = model_bundle_runtime.load_nn_architecture_args_from_bundle
    original_factory = model_bundle_runtime._create_nn_state_eval_from_existing_model

    def fake_load_nn_architecture_args_from_bundle(
        bundle_arg: ResolvedModelBundle,
    ) -> object:
        assert bundle_arg == bundle
        return architecture_args

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

    model_bundle_runtime.load_nn_architecture_args_from_bundle = (
        fake_load_nn_architecture_args_from_bundle
    )
    model_bundle_runtime._create_nn_state_eval_from_existing_model = (
        fake_create_nn_state_eval_from_existing_model
    )
    try:
        built = model_bundle_runtime.create_nn_state_eval_from_model_bundle_and_converter(
            bundle,
            content_to_input,
        )
    finally:
        model_bundle_runtime.load_nn_architecture_args_from_bundle = (
            original_bundle_loader
        )
        model_bundle_runtime._create_nn_state_eval_from_existing_model = original_factory

    assert built is evaluator
    assert calls == {
        "model_weights_file_name": "/tmp/model-bundle/weights.pt",
        "nn_architecture_args": architecture_args,
        "content_to_input_convert": content_to_input,
    }
