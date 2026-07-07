"""Tests for routing legacy chess trainer wrappers through common learning."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from chipiron.learning.supervised import (
    RegressionBatchMetricSums,
    RegressionBatchTrainStats,
    TensorSupervisedBatch,
    evaluate_regression_batch,
    train_regression_batch,
)


def test_common_train_regression_batch_accepts_fen_and_value_data() -> None:
    """The common train kernel should work with the chess batch object."""
    _require_chess_learning_dependencies()
    datasets = importlib.import_module(
        "chipiron.environments.chess.players.evaluators.boardevaluators."
        "datasets.datasets"
    )
    sample_1 = datasets.FenAndValueData(
        fen_tensor=torch.ones(5),
        value_tensor=torch.tensor([0.5], dtype=torch.float32),
    )
    sample_2 = datasets.FenAndValueData(
        fen_tensor=-torch.ones(5),
        value_tensor=torch.tensor([-0.5], dtype=torch.float32),
    )
    batch = datasets.custom_collate_fn_fen_and_value([sample_1, sample_2])

    model = nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    stats = train_regression_batch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch=batch,
        device=torch.device("cpu"),
        timings=None,
    )
    after = list(model.parameters())

    assert stats.target_count == 2
    assert stats.squared_error_sum >= 0.0
    assert any(
        not torch.equal(old, new) for old, new in zip(before, after, strict=True)
    )


def test_common_evaluate_regression_batch_accepts_fen_and_value_data() -> None:
    """The common eval kernel should work with the chess batch object."""
    _require_chess_learning_dependencies()
    datasets = importlib.import_module(
        "chipiron.environments.chess.players.evaluators.boardevaluators."
        "datasets.datasets"
    )
    sample_1 = datasets.FenAndValueData(
        fen_tensor=torch.ones(5),
        value_tensor=torch.tensor([0.5], dtype=torch.float32),
    )
    sample_2 = datasets.FenAndValueData(
        fen_tensor=-torch.ones(5),
        value_tensor=torch.tensor([-0.5], dtype=torch.float32),
    )
    batch = datasets.custom_collate_fn_fen_and_value([sample_1, sample_2])

    metrics = evaluate_regression_batch(
        model=nn.Linear(5, 1),
        batch=batch,
        device=torch.device("cpu"),
        timings=None,
    )

    assert metrics.target_count == 2
    assert metrics.squared_error_sum >= 0.0


def test_legacy_trainer_train_delegates_to_common_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy trainer train method should call the common train kernel."""
    nn_trainer_module = _import_legacy_trainer_module()
    model = nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = nn_trainer_module.NNPytorchTrainer(
        net=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    calls: list[dict[str, Any]] = []

    def fake_train_regression_batch(**kwargs: Any) -> RegressionBatchTrainStats:
        calls.append(kwargs)
        return RegressionBatchTrainStats(
            loss=1.25,
            squared_error_sum=2.5,
            absolute_error_sum=1.5,
            sample_count=2,
            target_count=2,
        )

    monkeypatch.setattr(
        nn_trainer_module,
        "train_regression_batch",
        fake_train_regression_batch,
    )

    loss = trainer.train(
        input_layer=torch.ones((2, 5)),
        target_value=torch.tensor([[0.5], [-0.5]], dtype=torch.float32),
    )

    assert calls
    assert calls[0]["model"] is model
    assert calls[0]["optimizer"] is optimizer
    assert isinstance(calls[0]["batch"], TensorSupervisedBatch)
    assert float(loss.cpu()) == pytest.approx(1.25)


def test_legacy_compute_test_error_delegates_to_common_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy dataset evaluator should call the common eval kernel."""
    nn_trainer_module = _import_legacy_trainer_module()
    model = nn.Linear(5, 1)
    calls: list[dict[str, Any]] = []

    def fake_evaluate_regression_batch(**kwargs: Any) -> RegressionBatchMetricSums:
        calls.append(kwargs)
        return RegressionBatchMetricSums(
            squared_error_sum=4.0,
            absolute_error_sum=2.0,
            target_count=2,
        )

    monkeypatch.setattr(
        nn_trainer_module,
        "evaluate_regression_batch",
        fake_evaluate_regression_batch,
    )
    data_loader = DataLoader(
        [
            TensorSupervisedBatch(
                input_tensor=torch.ones(5),
                target_tensor=torch.tensor([0.5], dtype=torch.float32),
                is_batch=False,
            )
        ],
        batch_size=1,
        collate_fn=_collate_tensor_supervised_batches,
    )

    test_error = nn_trainer_module.compute_test_error_on_dataset(
        net=model,
        criterion=nn.MSELoss(),
        data_test=data_loader,
        number_of_tests=3,
    )

    assert len(calls) == 3
    assert calls[0]["model"] is model
    assert isinstance(calls[0]["batch"], TensorSupervisedBatch)
    assert test_error == pytest.approx(2.0)


def _collate_tensor_supervised_batches(
    samples: list[TensorSupervisedBatch],
) -> TensorSupervisedBatch:
    """Collate concrete tensor supervised samples into one batch."""
    return TensorSupervisedBatch(
        input_tensor=torch.stack([sample.get_input_layer() for sample in samples]),
        target_tensor=torch.stack([sample.get_target_value() for sample in samples]),
        is_batch=True,
    )


def _import_legacy_trainer_module() -> Any:
    """Import the legacy trainer module when its optional dependencies exist."""
    pytest.importorskip("coral")
    return importlib.import_module("chipiron.learningprocesses.nn_trainer.nn_trainer")


def _require_chess_learning_dependencies() -> None:
    """Skip chess batch tests when optional runtime dependencies are absent."""
    pytest.importorskip("atomheart")
    pytest.importorskip("coral")
    pytest.importorskip("parsley")
