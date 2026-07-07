"""Import and adapter smoke tests for current chess learning modules."""

from __future__ import annotations

import importlib
import sys
import types
from types import ModuleType

import pytest
import torch

from chipiron.learning.supervised import (
    infer_batch_sample_count,
    move_supervised_batch_to_device,
)


def test_legacy_chess_trainer_package_imports() -> None:
    """The legacy chess trainer package should remain importable."""
    nn_trainer = importlib.import_module("chipiron.learningprocesses.nn_trainer")

    assert isinstance(nn_trainer, ModuleType)


def test_legacy_chess_trainer_modules_import() -> None:
    """Legacy factory module should keep public config and checkpoint exports."""
    _require_chess_learning_dependencies()

    factory = importlib.import_module("chipiron.learningprocesses.nn_trainer.factory")
    training_args = importlib.import_module(
        "chipiron.scripts.learn_nn_supervised.training_args"
    )

    assert training_args.SupervisedTrainingArgs is not None
    assert training_args.NNTrainerArgs is training_args.SupervisedTrainingArgs
    assert (
        training_args.NNTrainerConfigError
        is training_args.SupervisedTrainingConfigError
    )
    assert training_args.OptimizerType is not None
    assert factory.SupervisedTrainingArgs is training_args.SupervisedTrainingArgs
    assert factory.NNTrainerArgs is training_args.SupervisedTrainingArgs
    assert (
        factory.SupervisedTrainingConfigError
        is training_args.SupervisedTrainingConfigError
    )
    assert factory.NNTrainerConfigError is training_args.SupervisedTrainingConfigError
    assert factory.OptimizerType is training_args.OptimizerType
    assert factory.safe_nn_architecture_save is not None
    assert factory.safe_nn_param_save is not None
    assert factory.safe_nn_trainer_save is not None


def test_supervised_training_args_legacy_alias_preserves_validation() -> None:
    """The legacy config alias should preserve validation behavior."""
    _require_chess_learning_dependencies()
    training_args = importlib.import_module(
        "chipiron.scripts.learn_nn_supervised.training_args"
    )

    assert training_args.NNTrainerArgs is training_args.SupervisedTrainingArgs

    with pytest.raises(training_args.SupervisedTrainingConfigError):
        training_args.SupervisedTrainingArgs(reuse_existing_model=True)

    with pytest.raises(training_args.NNTrainerConfigError):
        training_args.NNTrainerArgs(reuse_existing_model=True)


def test_chess_learning_entrypoint_modules_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current chess learning and evaluation script modules should import."""
    _require_chess_learning_dependencies()
    _install_observability_stubs(monkeypatch)

    supervised_script = importlib.import_module(
        "chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets"
    )
    scratch_script = importlib.import_module(
        "chipiron.scripts.learn_from_scratch_value_and_fixed_boards."
        "learn_from_scratch_value_and_fixed_boards"
    )
    evaluation_script = importlib.import_module(
        "chipiron.scripts.evaluate_models.evaluate_models"
    )

    assert supervised_script.LearnNNScript is not None
    assert scratch_script.LearnNNFromScratchScript is not None
    assert evaluation_script.evaluate_models is not None


def test_chess_neural_network_evaluator_modules_import() -> None:
    """Chess neural evaluator configuration and runtime modules should import."""
    _require_chess_learning_dependencies()

    generic_neural_networks = importlib.import_module(
        "chipiron.players.boardevaluators.neural_networks"
    )
    chipiron_nn_args = importlib.import_module(
        "chipiron.environments.chess.players.evaluators.boardevaluators."
        "neural_networks.chipiron_nn_args"
    )
    chess_bundle_evaluator = importlib.import_module(
        "chipiron.environments.chess.players.evaluators.boardevaluators."
        "neural_networks.chess_model_bundle_evaluator"
    )

    assert generic_neural_networks.NeuralNetBoardEvalArgs is not None
    assert chipiron_nn_args.ChipironNNArgs is not None
    assert chess_bundle_evaluator.create_chess_nn_state_eval_from_model_bundle


def test_chess_fen_and_value_batch_uses_common_supervised_helpers() -> None:
    """A chess supervised batch object should satisfy common batch helpers."""
    _require_chess_learning_dependencies()

    datasets = importlib.import_module(
        "chipiron.environments.chess.players.evaluators.boardevaluators."
        "datasets.datasets"
    )
    sample = datasets.FenAndValueData(
        fen_tensor=torch.ones(5),
        value_tensor=torch.tensor([0.5]),
    )

    moved_sample = move_supervised_batch_to_device(sample, torch.device("cpu"))
    collated_batch = datasets.custom_collate_fn_fen_and_value([sample, sample])

    assert moved_sample.get_input_layer().device.type == "cpu"
    assert moved_sample.get_target_value().device.type == "cpu"
    assert infer_batch_sample_count(sample) == 1
    assert infer_batch_sample_count(collated_batch) == 2


def _require_chess_learning_dependencies() -> None:
    """Skip chess import smoke tests when optional runtime dependencies are absent."""
    pytest.importorskip("atomheart")
    pytest.importorskip("coral")


def _install_observability_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install tiny mlflow and torchinfo stubs for dependency-light imports."""
    mlflow_module = types.ModuleType("mlflow")
    mlflow_pytorch_module = types.ModuleType("mlflow.pytorch")
    mlflow_models_module = types.ModuleType("mlflow.models")
    mlflow_signature_module = types.ModuleType("mlflow.models.signature")
    torchinfo_module = types.ModuleType("torchinfo")

    mlflow_module.set_tracking_uri = lambda uri: None
    mlflow_module.log_metric = lambda *args, **kwargs: None
    mlflow_module.log_params = lambda params: None
    mlflow_module.log_artifact = lambda path: None
    mlflow_pytorch_module.log_model = lambda *args, **kwargs: None
    mlflow_pytorch_module.get_default_conda_env = dict
    mlflow_module.pytorch = mlflow_pytorch_module

    class _ModelSignature:
        """Placeholder signature type used by the supervised script."""

    mlflow_signature_module.ModelSignature = _ModelSignature
    mlflow_signature_module.infer_signature = lambda *args, **kwargs: _ModelSignature()
    mlflow_models_module.signature = mlflow_signature_module
    torchinfo_module.summary = lambda model: f"summary({type(model).__name__})"

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.pytorch", mlflow_pytorch_module)
    monkeypatch.setitem(sys.modules, "mlflow.models", mlflow_models_module)
    monkeypatch.setitem(sys.modules, "mlflow.models.signature", mlflow_signature_module)
    monkeypatch.setitem(sys.modules, "torchinfo", torchinfo_module)
