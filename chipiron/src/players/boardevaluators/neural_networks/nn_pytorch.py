import torch
import torch.nn as nn
import chess


class BoardNet(nn.Module):
    def __init__(self, path_to_main_folder, relative_path_file):
        super(BoardNet, self).__init__()
        self.path_to_param_file = path_to_main_folder + 'chipiron/runs/players/board_evaluators/nn_pytorch/' \
                                  + relative_path_file

    def load_or_init_weights(self):
        print('load_or_init_weights')
        try:  # load
            with open(self.path_to_param_file, 'rb') as fileNNR:
                print('loading the existing param file')
                self.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:  # init
            print('creating a new param file')
            with open(self.path_to_param_file, 'wb') as fileNNW:
                with torch.no_grad():
                    self.init_weights(fileNNW)

    def safe_save(self):
        try:
            with open(self.path_to_param_file, 'wb') as fileNNW:
                torch.save(self.state_dict(), fileNNW)
        except KeyboardInterrupt:
            with open(self.path_to_param_file, 'wb') as fileNNW:
                torch.save(self.state_dict(), fileNNW)
            exit(-1)






