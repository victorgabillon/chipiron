"""
This module defines the NetP1 class, which is a neural network model for board evaluation in a game.
"""

import torch
import torch.nn as nn

from chipiron.players.boardevaluators.neural_networks.board_to_tensor import transform_board_pieces_one_side
from chipiron.utils.chi_nn import ChiNN


class NetP1(ChiNN):
    """The NetP1 class is a subclass of ChiNN and implements the forward pass of the neural network. It consists of
     a single fully connected layer with a tanh activation function."""

    def __init__(self) -> None:
        """Constructor for the NetP1 class. Initializes the neural network layers."""
        super(NetP1, self).__init__()

        self.transform_board_function = transform_board_pieces_one_side
        self.fc1 = nn.Linear(5, 1)
        self.tanh = nn.Tanh()

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the neural network model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file: str) -> None:
        """
        Initialize the weights of the neural network model.

        Args:
            file (str): File path to save the initialized weights.
        """
        ran = torch.rand(5) * 0.001 + 0.03
        ran = ran.unsqueeze(0)
        self.fc1.weight = torch.nn.Parameter(ran)
        nn.init.constant(self.fc1.bias, 0.0)
        torch.save(self.state_dict(), file)
        for param in self.parameters():
            print(param.data)

    def print_param(self) -> None:
        """
        Print the parameters of the neural network model.
        """
        for layer, param in enumerate(self.parameters()):
            if layer == 0:
                print('pawns', param.data[0, 0])
                print('knights', param.data[0, 1])
                print('bishops', param.data[0, 2])
                print('rook', param.data[0, 3])
                print('queen', param.data[0, 4])
            else:
                print(param.data)

    def print_input(self, input: torch.Tensor) -> None:
        """
        Print the input tensor.

        Args:
            input (torch.Tensor): Input tensor.
        """
        print('pawns', input[0])
        print('knights', input[1])
        print('bishops', input[2])
        print('rook', input[3])
        print('queen', input[4])
