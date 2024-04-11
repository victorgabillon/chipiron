"""
This module defines the neural network model NetPP1, which is a subclass of ChiNN.
NetPP1 is used for board evaluation in the game of chess.

Classes:
- NetPP1: Neural network model for board evaluation.

Functions:
- print_input: Prints the input parameters of the neural network.
- print_piece_param: Prints the parameters of each piece in the neural network.
- print_input_param: Prints the input parameters of each piece in the neural network.
"""

import torch
import torch.nn as nn

from chipiron.utils.chi_nn import ChiNN


class NetPP1(ChiNN):
    """The NetPP1 class. Inherits from the ChiNN base class."""

    def __init__(self) -> None:
        """Constructor for the NetPP1 class."""
        super(NetPP1, self).__init__()

        self.transform_board_function = None
        self.fc1 = nn.Linear(384 + 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
        - x: Input tensor.

        Returns:
        - Output tensor.
        """
        x = self.fc1(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file: str) -> None:
        """
        Initializes the weights of the neural network.

        Args:
        - file: File path to save the initialized weights.
        """
        ran = torch.rand(384 + 2) * 0.001 + 0.03
        ran = ran.unsqueeze(0)
        self.fc1.weight = torch.nn.Parameter(ran)
        nn.init.constant(self.fc1.bias, 0.0)
        torch.save(self.state_dict(), file)
        for param in self.parameters():
            print(param.data)

    def print_param(self) -> None:
        """
        Prints the parameters of the neural network.
        """
        for layer, param in enumerate(self.parameters()):
            if layer == 0:
                print('pawns', sum(param.data[0, 64 * 0 + 8: 64 * 0 + 64 - 8]) / (64. - 16.))
                print_piece_param(0, param.data)
                print('knights', sum(param.data[0, 64 * 1: 64 * 1 + 64]) / 64.)
                print_piece_param(1, param.data)
                print('bishops', sum(param.data[0, 64 * 2: 64 * 2 + 64]) / 64.)
                print_piece_param(2, param.data)
                print('rook', sum(param.data[0, 64 * 3: 64 * 3 + 64]) / 64.)
                print_piece_param(3, param.data)
                print('queen', sum(param.data[0, 64 * 4: 64 * 4 + 64]) / 64.)
                print_piece_param(4, param.data)
                print('king', sum(param.data[0, 64 * 5: 64 * 5 + 64]) / 64.)
                print_piece_param(5, param.data)
                print('castling', )
                print(param.data[0, 64 * 6:64 * 6 + 2])
            else:
                print(param.data)


def print_input(input: torch.Tensor) -> None:
    """
    Prints the input parameters of the neural network.

    Args:
    - input: Input tensor.
    """
    print('pawns', sum(input[64 * 0 + 8: 64 * 0 + 64 - 8]) / (64. - 16.))
    print_input_param(0, input)
    print('knights', sum(input[64 * 1: 64 * 1 + 64]) / 64.)
    print_input_param(1, input)
    print('bishops', sum(input[64 * 2: 64 * 2 + 64]) / 64.)
    print_input_param(2, input)
    print('rook', sum(input[64 * 3: 64 * 3 + 64]) / 64.)
    print_input_param(3, input)
    print('queen', sum(input[64 * 4: 64 * 4 + 64]) / 64.)
    print_input_param(4, input)
    print('king', sum(input[64 * 5: 64 * 5 + 64]) / 64.)
    print_input_param(5, input)


def print_piece_param(i: int, vec: torch.Tensor) -> None:
    """
    Prints the parameters of each piece in the neural network.

    Args:
    - i: Index of the piece.
    - vec: Tensor containing the parameters.
    """
    for r in range(8):
        print(vec[0, 64 * i + 8 * r: 64 * i + 8 * (r + 1)])


def print_input_param(i: int, vec: torch.Tensor) -> None:
    """
    Prints the input parameters of each piece in the neural network.

    Args:
    - i: Index of the piece.
    - vec: Tensor containing the parameters.
    """
    for r in range(8):
        print(vec[64 * i + 8 * r: 64 * i + 8 * (r + 1)])
