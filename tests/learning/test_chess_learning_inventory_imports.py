"""Import and adapter smoke tests for current chess learning modules."""

from __future__ import annotations

import importlib
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
    """Important legacy trainer implementation modules should still import."""
    _require_chess_learning_dependencies()

    factory = importlib.import_module("chipiron.learningprocesses.nn_trainer.factory")
    trainer = importlib.import_module(
        "chipiron.learningprocesses.nn_trainer.nn_trainer"
    )

    assert factory.NNTrainerArgs is not None
    assert factory.create_nn_trainer is not None
    assert trainer.NNPytorchTrainer is not None
    assert trainer.compute_test_error_on_dataset is not None


def test_chess_learning_entrypoint_modules_import() -> None:
    """Current chess learning and evaluation script modules should import."""
    _require_chess_learning_dependencies()
    pytest.importorskip("PySide6")

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
    pytest.importorskip("parsley")
