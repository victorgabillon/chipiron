"""Document the module contains the definition of the NNPytorchTrainer class, which is responsible for training and testing a neural network model using PyTorch."""

import typing
from collections.abc import Callable

import torch
from coral.chi_nn import ChiNN
from torch.utils.data import DataLoader

from chipiron.learning import module_device
from chipiron.learning.supervised import (
    RegressionBatchMetricSums,
    SupervisedBatch,
    TensorSupervisedBatch,
    evaluate_regression_batch,
    train_regression_batch,
)
from chipiron.utils.logger import chipiron_logger


def compute_test_error_on_dataset(
    net: ChiNN,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_test: DataLoader[SupervisedBatch],
    number_of_tests: int = 100,
) -> float:
    """Compute the test error on the given dataset.

    Args:
        net (ChiNN): The neural network model to be tested.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function used for testing.
        data_test (DataLoader[SupervisedBatch]): The test dataset.
        number_of_tests (int, optional): The number of tests to run. Defaults to 100.

    Returns:
        float: The test error on the given dataset.

    """
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    target_count = 0
    device = module_device(net)
    for _ in range(number_of_tests):
        sample = next(iter(data_test))
        batch_metrics = evaluate_regression_batch(
            model=net,
            batch=sample,
            device=device,
            timings=None,
        )
        squared_error_sum += batch_metrics.squared_error_sum
        absolute_error_sum += batch_metrics.absolute_error_sum
        target_count += batch_metrics.target_count

    test_error = _loss_value_from_regression_sums(
        criterion=criterion,
        metrics=RegressionBatchMetricSums(
            squared_error_sum=squared_error_sum,
            absolute_error_sum=absolute_error_sum,
            target_count=target_count,
        ),
    )
    chipiron_logger.info("test error %f", test_error)
    return test_error


class NNPytorchTrainer:
    """A class that trains a neural network model using PyTorch.

    Args:
        net (ChiNN): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.

    Attributes:
        net (ChiNN): The neural network model to be trained.
        criterion (torch.nn.L1Loss): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.

    Methods:
        train(input_layer, target_value): Trains the neural network model using the provided input and target values.
        test(input_layer, target_value): Tests the neural network model using the provided input and target values.
        train_next_boards(input_layer, next_input_layer): Trains the neural network model using the provided input and next input layers.

    """

    def __init__(
        self,
        net: ChiNN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Initialize a new instance of the NNPytorchTrainer class.

        Args:
            net (ChiNN): The neural network model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.

        Returns:
            None

        """
        self.net = net
        self.criterion = torch.nn.L1Loss()
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        chipiron_logger.info("Model put to device %s", self.device)

    @typing.no_type_check
    def train(
        self, input_layer: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        """Train the neural network model using the provided input and target values.

        Args:
            input_layer (torch.Tensor): The input data.
            target_value (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The loss value.

        """
        self.net.train()

        batch = TensorSupervisedBatch(
            input_tensor=input_layer,
            target_tensor=target_value,
            is_batch=True,
        )
        batch_stats = train_regression_batch(
            model=self.net,
            optimizer=self.optimizer,
            criterion=self.criterion,
            batch=batch,
            device=self.device,
            timings=None,
        )
        return torch.tensor(batch_stats.loss, device=self.device)

    def test(
        self, input_layer: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        """Test the neural network model using the provided input and target values.

        Args:
            input_layer (torch.Tensor): The input data.
            target_value (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The loss value.

        """
        self.net.eval()
        batch = TensorSupervisedBatch(
            input_tensor=input_layer,
            target_tensor=target_value,
            is_batch=True,
        )
        batch_metrics = evaluate_regression_batch(
            model=self.net,
            batch=batch,
            device=self.device,
            timings=None,
        )
        loss = torch.tensor(
            _loss_value_from_regression_sums(
                criterion=self.criterion,
                metrics=batch_metrics,
            ),
            device=self.device,
        )
        self.net.train()
        return loss

    def train_next_boards(
        self, input_layer: torch.Tensor, next_input_layer: torch.Tensor
    ) -> None:
        """Train the neural network model using the provided input and next input layers.

        Args:
            input_layer (torch.Tensor): The input data.
            next_input_layer (torch.Tensor): The next input data.

        Returns:
            None

        """
        # TODO(PR15): confirm whether this unreferenced legacy special case can be removed.
        self.net.eval()
        target_value = -self.net(next_input_layer)

        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)

        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        loss.backward()
        self.optimizer.step()

    def compute_test_error_on_dataset(
        self, data_test: DataLoader[SupervisedBatch]
    ) -> float:
        """Compute the test error of the neural network model.

        This method iterates over a test dataset and calculates the average loss
        for a given number of iterations. The test error is then computed as the
        average loss divided by the number of iterations.

        Returns:
            None

        """
        self.net.eval()
        test_error: float = compute_test_error_on_dataset(
            net=self.net, criterion=self.criterion, data_test=data_test
        )
        self.net.train()
        return test_error


def _loss_value_from_regression_sums(
    *,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: RegressionBatchMetricSums,
) -> float:
    """Convert common regression sums back to the legacy criterion scalar."""
    if metrics.target_count == 0:
        return 0.0
    if isinstance(criterion, torch.nn.MSELoss):
        if criterion.reduction == "sum":
            return metrics.squared_error_sum
        if criterion.reduction == "mean":
            return metrics.squared_error_sum / metrics.target_count
        if criterion.reduction == "none":
            raise _unreduced_loss_reconstruction_error("MSELoss")
    if isinstance(criterion, torch.nn.L1Loss):
        if criterion.reduction == "sum":
            return metrics.absolute_error_sum
        if criterion.reduction == "mean":
            return metrics.absolute_error_sum / metrics.target_count
        if criterion.reduction == "none":
            raise _unreduced_loss_reconstruction_error("L1Loss")
    raise _unsupported_loss_reconstruction_error()


def _unreduced_loss_reconstruction_error(loss_name: str) -> TypeError:
    """Return a clear error for unreduced criterion reconstruction."""
    return TypeError(
        f"Cannot reconstruct unreduced {loss_name} from aggregate regression metrics."
    )


def _unsupported_loss_reconstruction_error() -> TypeError:
    """Return a clear error for unsupported criterion reconstruction."""
    return TypeError(
        "Legacy chess trainer can only reconstruct scalar losses for MSELoss or "
        "L1Loss from common regression metric sums."
    )
