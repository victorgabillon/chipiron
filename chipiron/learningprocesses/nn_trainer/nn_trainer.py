"""
This module contains the definition of the NNPytorchTrainer class, which is responsible for training and testing a neural network model using PyTorch.
"""

import typing

import torch

from chipiron.utils.chi_nn import ChiNN


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
        prediction_with_player_to_move_as_white = self.net(input_layer)
        loss: torch.Tensor = self.criterion(
            prediction_with_player_to_move_as_white, target_value
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
        prediction_with_player_to_move_as_white = self.net(input_layer)
        loss: torch.Tensor = self.criterion(
            prediction_with_player_to_move_as_white, target_value
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
