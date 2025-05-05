"""
This module contains the definition of the NNPytorchTrainer class, which is responsible for training and testing a neural network model using PyTorch.
"""

import typing
from typing import Callable

import torch
from torch.utils.data import DataLoader

from chipiron.utils.chi_nn import ChiNN
from chipiron.utils.logger import chipiron_logger


def compute_loss(
    net: ChiNN,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    input_layer: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    prediction: torch.Tensor = net(input_layer)
    loss: torch.Tensor = criterion(prediction, target_value)
    return loss


def check_model_device(model: ChiNN) -> str | torch.device | int:

    # Check the device of the first parameter

    first_param_device = next(model.parameters()).device

    return first_param_device


def compute_test_error_on_dataset(
    net: ChiNN,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_test: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    number_of_tests: int = 100,
) -> float:
    sum_loss_test = 0.0
    count_test = 0
    loss_test: torch.Tensor
    device = check_model_device(net)
    for i in range(number_of_tests):
        sample_batched_test = next(iter(data_test))
        input_layer, target_value = sample_batched_test[0].to(
            device
        ), sample_batched_test[1].to(device)

        loss_test = compute_loss(
            net=net,
            criterion=criterion,
            input_layer=input_layer,
            target_value=target_value,
        )
        sum_loss_test += float(loss_test)
        count_test += 1
    chipiron_logger.info(f"test error {float(sum_loss_test / float(count_test))}")
    test_error: float = float(sum_loss_test / float(count_test))
    return test_error


class NNPytorchTrainer:
    """
    A class that trains a neural network model using PyTorch.

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
        """
        Initializes a new instance of the NNPytorchTrainer class.

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
        chipiron_logger.info(f"Model put to device {self.device}")

    @typing.no_type_check
    def train(
        self, input_layer: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Trains the neural network model using the provided input and target values.

        Args:
            input_layer (torch.Tensor): The input data.
            target_value (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The loss value.
        """
        self.net.train()

        self.optimizer.zero_grad()
        input_layer, target_value = input_layer.to(self.device), target_value.to(
            self.device
        )

        loss: torch.Tensor = compute_loss(
            net=self.net,
            criterion=self.criterion,
            input_layer=input_layer,
            target_value=target_value,
        )
        loss.backward()

        self.optimizer.step()
        return loss

    def test(
        self, input_layer: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Tests the neural network model using the provided input and target values.

        Args:
            input_layer (torch.Tensor): The input data.
            target_value (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The loss value.
        """
        self.net.eval()
        input_layer, target_value = input_layer.to(self.device), target_value.to(
            self.device
        )
        loss: torch.Tensor = compute_loss(
            net=self.net,
            criterion=self.criterion,
            input_layer=input_layer,
            target_value=target_value,
        )
        self.net.train()
        return loss

    def train_next_boards(
        self, input_layer: torch.Tensor, next_input_layer: torch.Tensor
    ) -> None:
        """
        Trains the neural network model using the provided input and next input layers.

        Args:
            input_layer (torch.Tensor): The input data.
            next_input_layer (torch.Tensor): The next input data.

        Returns:
            None
        """
        self.net.eval()
        target_value = -self.net(next_input_layer)

        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)

        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        loss.backward()
        self.optimizer.step()

    def compute_test_error_on_dataset(
        self, data_test: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Computes the test error of the neural network model.

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
