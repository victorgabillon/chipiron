from players.boardevaluators.neural_networks.models.nn_pp1 import NetPP1
from players.boardevaluators.neural_networks.models.nn_pp2 import NetPP2
from players.boardevaluators.neural_networks.models.nn_pp2d2 import NetPP2D2
from players.boardevaluators.neural_networks.models.nn_pp2d2_2 import NetPP2D2_2
from players.boardevaluators.neural_networks.models.nn_pp2d2_2leaky import NetPP2D2_2_LEAKY
from players.boardevaluators.neural_networks.models.nn_pp2d2_2prelu import NetPP2D2_2_PRELU
from players.boardevaluators.neural_networks.models.nn_pp2d2_2rrelu import NetPP2D2_2_RRELU
from chipiron.players.boardevaluators.neural_networks.board_evaluator import NNBoardEvaluator
from chipiron.extra_tools.small_tools import mkdir
from chipiron.extra_tools.chi_nn import ChiNN
import sys
from players.boardevaluators.neural_networks.output_converters.output_value_converter import OutputValueConverter,OneDToValueWhite
from players.boardevaluators.neural_networks.input_converters.board_to_input import Representation364BTI, BoardToInput
from players.boardevaluators.neural_networks.input_converters.factory import Representation364Factory


def get_folder_path_from(nn_type, nn_param_folder_name):
    folder_path = 'data/players/board_evaluators/nn_pytorch/nn_' + nn_type + '/' + nn_param_folder_name
    return folder_path


def get_nn_param_file_path_from(folder_path):
    nn_param_file_path = folder_path + '/param.pt'
    return nn_param_file_path


def create_nn(args, create_file=False) -> ChiNN:
    nn_type: str = args['nn_type']
    folder_path = get_folder_path_from(nn_type=nn_type, nn_param_folder_name=args['nn_param_folder_name'])
    mkdir(folder_path)
    net: ChiNN
    match nn_type:
        case 'pp1':
            net = NetPP1()
        case 'pp2':
            net = NetPP2()
        case 'pp2d2':
            net = NetPP2D2()
        case 'pp2d2_2':
            net = NetPP2D2_2()
        case 'pp2d2_2_leaky':
            net = NetPP2D2_2_LEAKY()
        case 'pp2d2_2_rrelu':
            net = NetPP2D2_2_RRELU()
        case 'pp2d2_2_prelu':
            net = NetPP2D2_2_PRELU()
        case other:
            sys.exit(f'Create NN: can not find {other}')

    nn_param_file_path = get_nn_param_file_path_from(folder_path)
    net.load_from_file_or_init_weights(nn_param_file_path, create_file)

    net.eval()
    return net


def create_nn_board_eval(arg, create_file=False) -> NNBoardEvaluator:
    net = create_nn(arg, create_file=create_file)
    output_and_value_converter: OutputValueConverter = OneDToValueWhite(point_of_view=net.evaluation_point_of_view)
    representation_factory :Representation364Factory = Representation364Factory()
    board_to_input_converter : BoardToInput = Representation364BTI(
        representation_factory= representation_factory
    )
    return NNBoardEvaluator(
        net=net,
        output_and_value_converter=output_and_value_converter,
        board_to_input_converter=board_to_input_converter
    )
