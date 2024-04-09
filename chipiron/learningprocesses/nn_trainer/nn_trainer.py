import typing

import torch

from chipiron.utils.chi_nn import ChiNN


class NNPytorchTrainer:

    def __init__(
            self,
            net: ChiNN,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler
    ) -> None:
        self.net = net
        self.criterion = torch.nn.L1Loss()
        self.optimizer = optimizer
        self.scheduler = scheduler

    @typing.no_type_check
    def train(
            self,
            input_layer: torch.Tensor,
            target_value: torch.Tensor
    ) -> torch.Tensor:
        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)
        loss: torch.Tensor = self.criterion(prediction_with_player_to_move_as_white, target_value)
        loss.backward()
        self.optimizer.step()
        return loss

    def test(
            self,
            input_layer: torch.Tensor,
            target_value: torch.Tensor
    ) -> torch.Tensor:
        self.net.eval()
        prediction_with_player_to_move_as_white = self.net(input_layer)
        loss: torch.Tensor = self.criterion(prediction_with_player_to_move_as_white, target_value)
        self.net.train()
        return loss

    def train_next_boards(
            self,
            input_layer: torch.Tensor,
            next_input_layer: torch.Tensor
    ) -> None:
        self.net.eval()
        target_value = - self.net(next_input_layer)

        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)

        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        loss.backward()
        self.optimizer.step()
