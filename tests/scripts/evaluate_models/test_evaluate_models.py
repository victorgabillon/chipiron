"""Module for test evaluate models."""

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("dacite")
pytest.importorskip("torch")

from chipiron.models.model_bundle import ModelBundleRef, ResolvedModelBundle
from chipiron.scripts.evaluate_models.evaluate_models import evaluate_models
from tests.model_bundle_test_utils import create_tiny_model_bundle


def test_evaluate_model_uses_model_bundle_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model evaluation should resolve from ModelBundleRef rather than legacy paths."""
    bundle_dir = create_tiny_model_bundle(
        tmp_path,
        bundle_name="evaluate_model_bundle",
        input_representation="piece_difference",
    )
    weights_file = bundle_dir / "weights.pt"
    bundle = ResolvedModelBundle(
        bundle_root=str(bundle_dir),
        weights_file_path=str(weights_file),
        architecture_file_path=str(bundle_dir / "architecture.yaml"),
        chipiron_nn_file_path=str(bundle_dir / "chipiron_nn.yaml"),
    )
    model_ref = ModelBundleRef(
        uri=str(bundle_dir),
        weights_file="weights.pt",
    )
    calls: dict[str, object] = {}

    class _FakeOutputConverter:
        def from_value_white_to_model_output(self, value: float) -> float:
            return value

    class _FakeNet:
        def parameters(self) -> list[object]:
            return []

    class _FakeEvaluator:
        def __init__(self) -> None:
            self.content_to_input_convert = lambda state: state
            self.output_and_value_converter = _FakeOutputConverter()
            self.net = _FakeNet()

    class _FakeDataset:
        def __init__(self, **kwargs: object) -> None:
            calls["dataset_kwargs"] = kwargs

        def load(self) -> None:
            calls["dataset_loaded"] = True

    def fake_create_chess_nn_state_eval_from_model_bundle(
        bundle_arg: ResolvedModelBundle,
    ) -> _FakeEvaluator:
        calls["evaluator_bundle"] = bundle_arg
        return _FakeEvaluator()

    class _FakeDataLoader:
        @classmethod
        def __class_getitem__(cls, item: object) -> type["_FakeDataLoader"]:
            _ = item
            return cls

        def __new__(cls, *args: object, **kwargs: object) -> list[int]:
            _ = cls
            calls["data_loader_args"] = args
            calls["data_loader_kwargs"] = kwargs
            return [1]

    monkeypatch.setattr(
        "chipiron.scripts.evaluate_models.evaluate_models.create_chess_nn_state_eval_from_model_bundle",
        fake_create_chess_nn_state_eval_from_model_bundle,
    )
    monkeypatch.setattr(
        "chipiron.scripts.evaluate_models.evaluate_models.FenAndValueDataSet",
        _FakeDataset,
    )
    monkeypatch.setattr(
        "chipiron.scripts.evaluate_models.evaluate_models.DataLoader",
        _FakeDataLoader,
    )
    monkeypatch.setattr(
        "chipiron.scripts.evaluate_models.evaluate_models.compute_test_error_on_dataset",
        lambda **kwargs: 0.125,
    )
    monkeypatch.setattr(
        "chipiron.scripts.evaluate_models.evaluate_models.count_parameters",
        lambda model: 7,
    )

    report_path = tmp_path / "test_evaluation_report.yaml"

    evaluate_models(
        models_to_evaluate=[model_ref],
        evaluation_report_file=str(report_path),
        dataset_file_name="dummy-dataset.pi",
    )

    assert calls["evaluator_bundle"] == bundle
    assert report_path.is_file()
