"""
Module for the ChiNN class
"""

import sys

import torch
import torch.nn as nn

from chipiron.utils import path
from chipiron.utils.logger import chipiron_logger


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
        chipiron_logger.info(f"load_or_init_weights from {path_to_param_file}")
        try:  # load
            with open(path_to_param_file, "rb") as fileNNR:
                chipiron_logger.info(
                    f"loading the existing param file {path_to_param_file}"
                )
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(fileNNR))
                else:
                    self.load_state_dict(
                        torch.load(fileNNR, map_location=torch.device("cpu"))
                    )

        except EnvironmentError:  # init
            chipiron_logger.error(f"no file {path_to_param_file}")
            sys.exit(
                "Error: no NN weights file and no rights to create it for file {}".format(
                    path_to_param_file,
                )
            )

    def log_readable_model_weights_to_file(self, file_path: str) -> None:
        raise Exception("not implemented in base class")
