import sys
import torch
import torch.nn as nn


class BoardNet(nn.Module):
    """
    The Generic Neural network class
    """

    def __init__(self):
        super(BoardNet, self).__init__()


    def load_from_file_or_init_weights(self, path_to_param_file, authorisation_to_create_file):
        print('load_or_init_weights')
        try:  # load
            with open(path_to_param_file, 'rb') as fileNNR:
                print('loading the existing param file')
                self.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:  # init
            print('no file', path_to_param_file)
            if authorisation_to_create_file:
                print('creating a new param file')
                with open(path_to_param_file, 'wb') as fileNNW:
                    with torch.no_grad():
                        self.init_weights(fileNNW)
            else:
                sys.exit(
                    'Error: no NN weights file and no rights to create it for file {} with authorisation {}'.format(
                        path_to_param_file, authorisation_to_create_file))

