from src.players.boardevaluators.neural_networks.nn_pp1 import NetPP1
from src.players.boardevaluators.neural_networks.nn_pp2 import NetPP2
from src.players.boardevaluators.neural_networks.nn_pp2d2 import NetPP2D2
from src.players.boardevaluators.neural_networks.nn_pp2d2_2 import NetPP2D2_2
from src.players.boardevaluators.neural_networks.nn_board_eval import NNBoardEval


def create_nn(arg, create_file=False):
    nn_type = arg['nn_type']
    if nn_type == 'pp1':
        net = NetPP1(arg['nn_param_file_name'])
    elif nn_type == 'pp2':
        net = NetPP2(arg['nn_param_file_name'])
    elif nn_type == 'pp2d2':
        net = NetPP2D2(arg['nn_param_file_name'])
    elif nn_type == 'pp2d2_2':
        net = NetPP2D2_2(arg['nn_param_file_name'])

    net.load_from_file_or_init_weights(create_file)
    net.eval()
    return net


def create_nn_board_eval(arg, create_file=False):
    net = create_nn(arg, create_file=create_file)
    return NNBoardEval(net)
