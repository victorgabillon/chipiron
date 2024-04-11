"""
This module defines the neural network model NetP2 for board evaluation in a game.

The NetP2 class inherits from the ChiNN class and implements the forward pass and weight initialization methods.

Attributes:
    transform_board_function (function): A function to transform the board pieces into a tensor representation.
    fc1 (torch.nn.Linear): The fully connected layer of the neural network.
    tanh (torch.nn.Tanh): The activation function.

Methods:
    forward(x: torch.Tensor) -> torch.Tensor: Performs the forward pass of the neural network.
    init_weights(file: str) -> None: Initializes the weights of the neural network and saves the state dictionary to a file.
    print_param() -> None: Prints the parameters of the neural network.
    print_input(input: torch.Tensor) -> None: Prints the input tensor.

"""

import torch
import torch.nn as nn

from chipiron.players.boardevaluators.neural_networks.board_to_tensor import transform_board_pieces_two_sides
from chipiron.utils.chi_nn import ChiNN


class NetP2(ChiNN):
    """The NetP2 class. Inherits from the ChiNN base class."""
    def __init__(
            self
    ) -> None:
        """Constructor for the NetP2 class. Initializes the neural network layers."""
        super(NetP2, self).__init__()

        self.transform_board_function = transform_board_pieces_two_sides
        self.fc1 = nn.Linear(10, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.fc1(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file: str) -> None:
        """
        Initializes the weights of the neural network and saves the state dictionary to a file.

        Args:
            file (str): The file path to save the state dictionary.
        """
        ran = torch.rand(10) * 0.001 + 0.03
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
                print('pawns', param.data[0, 0])
                print('knights', param.data[0, 1])
                print('bishops', param.data[0, 2])
                print('rook', param.data[0, 3])
                print('queen', param.data[0, 4])
                # print('king', param.data[0, 5])
                print('pawns-opp', param.data[0, 5])
                print('knights-opp', param.data[0, 6])
                print('bishops-opp', param.data[0, 7])
                print('rook-opp', param.data[0, 8])
                print('queen-opp', param.data[0, 9])
            else:
                print(param.data)

    def print_input(self, input: torch.Tensor) -> None:
        """
        Prints the input tensor.

        Args:
            input (torch.Tensor): The input tensor.
        """
        print('pawns', input[0])
        print('knights', input[1])
        print('bishops', input[2])
        print('rook', input[3])
        print('queen', input[4])
        #  print('king', input[5])
        print('pawns-opp', input[5])
        print('knights-opp', input[6])
        print('bishops-opp', input[7])
        print('rook-opp', input[8])
        print('queen-opp', input[9])
