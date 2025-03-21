"""
Module for the ChiNN class
"""

import sys

import torch
import torch.nn as nn

from chipiron.utils import path


class ChiNN(nn.Module):
    """
    The Generic Neural network class of chipiron
    """

    def __init__(self) -> None:
        """
        Initializes an instance of the ChiNN class.
        """
        super(ChiNN, self).__init__()

    def __getstate__(self) -> None:
        """
        Get the state of the object for pickling.

        Returns:
            None
        """
        return None

    def init_weights(self) -> None:
        """
        Initialize the weights of the model.
        """
        pass

    def load_weights_from_file(self, path_to_param_file: path) -> None:
        """
        Loads the neural network weights from a file or initializes them if the file doesn't exist.

        Args:
            path_to_param_file (str): The path to the parameter file.
            authorisation_to_create_file (bool): Flag indicating whether the program has authorization to create a new file.

        Returns:
            None
        """
        print(f"load_or_init_weights from {path_to_param_file}")
        try:  # load
            with open(path_to_param_file, "rb") as fileNNR:
                print("loading the existing param file", path_to_param_file)
                self.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:  # init
            print("no file", path_to_param_file)
            sys.exit(
                "Error: no NN weights file and no rights to create it for file {}".format(
                    path_to_param_file,
                )
            )
