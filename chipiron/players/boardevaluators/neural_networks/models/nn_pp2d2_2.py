"""
This module defines the neural network model NetPP2D2_2, which is a subclass of ChiNN.
"""

from typing import Any

import torch
import torch.nn as nn

from chipiron.players.boardevaluators.board_evaluation.board_evaluation import PointOfView
from chipiron.utils.chi_nn import ChiNN


class NetPP2D2_2(ChiNN):
    """The NetPP2D2_2 class. Inherits from the ChiNN base class."""
    def __init__(self) -> None:
        """Constructor for the NetPP2D2_2 class."""
        super(NetPP2D2_2, self).__init__()
        self.evaluation_point_of_view = PointOfView.PLAYER_TO_MOVE

        self.fc1 = nn.Linear(772, 20)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.relu_1(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file: str) -> None:
        """
        Initialize the weights of the neural network.

        Args:
            file (str): File path to load the weights from.
        """
        pass

    def print_param(self) -> None:
        """
        Print the parameters of the neural network.
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
                print('pawns-opposite', sum(param.data[0, 64 * 6 + 8: 64 * 6 + 64 - 8]) / (64. - 16.))
                print_piece_param(6, param.data)
                print('knights-opposite', sum(param.data[0, 64 * 7: 64 * 7 + 64]) / 64.)
                print_piece_param(7, param.data)
                print('bishops-opposite', sum(param.data[0, 64 * 8: 64 * 8 + 64]) / 64.)
                print_piece_param(8, param.data)
                print('rook-opposite', sum(param.data[0, 64 * 9: 64 * 9 + 64]) / 64.)
                print_piece_param(9, param.data)
                print('queen-opposite', sum(param.data[0, 64 * 10: 64 * 10 + 64]) / 64.)
                print_piece_param(10, param.data)
                print('king-opposite', sum(param.data[0, 64 * 11: 64 * 11 + 64]) / 64.)
                print_piece_param(11, param.data)
                print('castling', param.data[0, 64 * 12: 64 * 12 + 2])
                print('castlingopposite', param.data[0, 64 * 12 + 2: 64 * 12 + 4])
            else:
                print('other layer', layer, param.data)

    def print_input(self, input: torch.Tensor) -> None:
        """
        Print the input tensor.

        Args:
            input (torch.Tensor): Input tensor.
        """
        print('pawns', sum(input[0, 64 * 0 + 8: 64 * 0 + 64 - 8]) / (64. - 16.))
        print_piece_param(0, input)
        print('knights', sum(input[0, 64 * 1: 64 * 1 + 64]) / 64.)
        print_piece_param(1, input)
        print('bishops', sum(input[0, 64 * 2: 64 * 2 + 64]) / 64.)
        print_piece_param(2, input)
        print('rook', sum(input[0, 64 * 3: 64 * 3 + 64]) / 64.)
        print_piece_param(3, input)
        print('queen', sum(input[0, 64 * 4: 64 * 4 + 64]) / 64.)
        print_piece_param(4, input)
        print('king', sum(input[0, 64 * 5: 64 * 5 + 64]) / 64.)
        print_piece_param(5, input)
        print('pawns-opposite', sum(input[0, 64 * 6 + 8: 64 * 6 + 64 - 8]) / (64. - 16.))
        print_piece_param(6, input)
        print('knights-opposite', sum(input[0, 64 * 7: 64 * 7 + 64]) / 64.)
        print_piece_param(7, input)
        print('bishops-opposite', sum(input[0, 64 * 8: 64 * 8 + 64]) / 64.)
        print_piece_param(8, input)
        print('rook-opposite', sum(input[0, 64 * 9: 64 * 9 + 64]) / 64.)
        print_piece_param(9, input)
        print('queen-opposite', sum(input[0, 64 * 10: 64 * 10 + 64]) / 64.)
        print_piece_param(10, input)
        print('king-opposite', sum(input[0, 64 * 11: 64 * 11 + 64]) / 64.)
        print_piece_param(11, input)

    def get_nn_input(self, node: Any) -> None:
        """
        Get the input tensor for the neural network.

        Args:
            node (Any): Node object.

        Raises:
            Exception: To be recoded in the current module.
        """
        raise Exception(f'to be recoded in {__name__}')


def print_piece_param(i: int, vec: torch.Tensor) -> None:
    """
    Print the parameters of a specific piece.

    Args:
        i (int): Index of the piece.
        vec (torch.Tensor): Tensor containing the parameters.
    """
    for r in range(8):
        print(vec[0, 64 * i + 8 * r: 64 * i + 8 * (r + 1)])
