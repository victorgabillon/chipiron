from src.players.boardevaluators.neural_networks.nn_pp1 import NetPP1
from src.players.boardevaluators.neural_networks.nn_pp2 import NetPP2
from src.players.boardevaluators.neural_networks.nn_pp2d2 import NetPP2D2
from src.players.boardevaluators.neural_networks.nn_pp2d2_2 import NetPP2D2_2
from src.players.boardevaluators.neural_networks.nn_pp2d2_2leaky import NetPP2D2_2_LEAKY
from src.players.boardevaluators.neural_networks.nn_pp2d2_2prelu import NetPP2D2_2_PRELU
from src.players.boardevaluators.neural_networks.nn_pp2d2_2rrelu import NetPP2D2_2_RRELU
from src.players.boardevaluators.neural_networks.nn_board_eval import NNBoardEval
from src.extra_tools.small_tools import mkdir


def get_folder_path_from(nn_type, nn_param_folder_name):
    folder_path = 'chipiron/data/players/board_evaluators/nn_pytorch/nn_' + nn_type + '/' + nn_param_folder_name
    return folder_path


def get_nn_param_file_path_from(folder_path):
    nn_param_file_path = folder_path + '/param.pt'
    return nn_param_file_path


def create_nn(args, create_file=False):
    nn_type = args['nn_type']
    folder_path = get_folder_path_from(nn_type=nn_type, nn_param_folder_name=args['nn_param_folder_name'])
    mkdir(folder_path)

    if nn_type == 'pp1':
        net = NetPP1()
    elif nn_type == 'pp2':
        net = NetPP2()
    elif nn_type == 'pp2d2':
        net = NetPP2D2()
    elif nn_type == 'pp2d2_2':
        net = NetPP2D2_2()
    elif nn_type == 'pp2d2_2_leaky':
        net = NetPP2D2_2_LEAKY()
    elif nn_type == 'pp2d2_2_rrelu':
        net = NetPP2D2_2_RRELU()
    elif nn_type == 'pp2d2_2_prelu':
        net = NetPP2D2_2_PRELU()

    nn_param_file_path = get_nn_param_file_path_from(folder_path)
    net.load_from_file_or_init_weights(nn_param_file_path, create_file)

    net.eval()
    return net


def create_nn_board_eval(arg, create_file=False):
    net = create_nn(arg, create_file=create_file)
    return NNBoardEval(net)
