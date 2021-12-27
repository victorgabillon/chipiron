import sys
import torch
import torch.nn as nn


class BoardNet(nn.Module):
    def __init__(self, relative_path_file):
        super(BoardNet, self).__init__()
        self.path_to_param_file = 'chipiron/data/players/board_evaluators/nn_pytorch/' \
                                  + relative_path_file

    def load_from_file_or_init_weights(self, authorisation_to_create_file):
        print('load_or_init_weights')
        try:  # load
            with open(self.path_to_param_file, 'rb') as fileNNR:
                print('loading the existing param file')
                self.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:  # init
            print('no file', self.path_to_param_file)
            if authorisation_to_create_file:
                print('creating a new param file')
                with open(self.path_to_param_file, 'wb') as fileNNW:
                    with torch.no_grad():
                        self.init_weights(fileNNW)
            else:
                sys.exit(
                    'Error: no NN weights file and no rights to create it for file {} with authorisation {}'.format(
                        self.path_to_param_file, authorisation_to_create_file))

    def safe_save(self):
        try:
            with open(self.path_to_param_file, 'wb') as fileNNW:
                torch.save(self.state_dict(), fileNNW)
            with open(self.path_to_param_file + '_save', 'wb') as fileNNW:
                torch.save(self.state_dict(), fileNNW)
        except KeyboardInterrupt:
            with open(self.path_to_param_file + '_save', 'wb') as fileNNW:
                torch.save(self.state_dict(), fileNNW)
            exit(-1)
