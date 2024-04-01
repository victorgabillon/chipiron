import os.path
import sys

from chipiron.players.boardevaluators.neural_networks import NeuralNetBoardEvalArgs
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import BoardToInput
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import Representation364Factory
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_364_bti import \
    Representation364BTI
from chipiron.players.boardevaluators.neural_networks.models.nn_pp1 import NetPP1
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2 import NetPP2
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2 import NetPP2D2
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2 import NetPP2D2_2
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2leaky import NetPP2D2_2_LEAKY
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2prelu import NetPP2D2_2_PRELU
from chipiron.players.boardevaluators.neural_networks.models.nn_pp2d2_2rrelu import NetPP2D2_2_RRELU
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import NNBoardEvaluator
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import \
    OutputValueConverter, OneDToValueWhite
from chipiron.utils import path
from chipiron.utils.chi_nn import ChiNN
from chipiron.utils.small_tools import mkdir


def get_folder_path_from(
        nn_type: str,
        nn_param_folder_name: str
) -> path:
    print('nn_type', nn_type)
    folder_path = os.path.join('data/players/board_evaluators/nn_pytorch/nn_' + nn_type, nn_param_folder_name)
    return folder_path


def get_nn_param_file_path_from(folder_path):
    nn_param_file_path = os.path.join(folder_path, 'param.pt')
    return nn_param_file_path


def create_nn(
        args: NeuralNetBoardEvalArgs,
        create_file=False
) -> ChiNN:
    folder_path = get_folder_path_from(nn_type=args.nn_type, nn_param_folder_name=args.nn_param_folder_name)
    mkdir(folder_path)

    net: ChiNN
    match args.nn_type:
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
            sys.exit(f'Create NN: can not find {other} in file {__name__}')

    nn_param_file_path = get_nn_param_file_path_from(folder_path)
    print('nn_param_file_path', nn_param_file_path, create_file)
    net.load_from_file_or_init_weights(nn_param_file_path, create_file)

    net.eval()
    return net


def create_nn_board_eval(
        arg: NeuralNetBoardEvalArgs,
        create_file=False
) -> NNBoardEvaluator:
    net = create_nn(arg, create_file=create_file)
    output_and_value_converter: OutputValueConverter = OneDToValueWhite(point_of_view=net.evaluation_point_of_view)
    representation_factory: Representation364Factory = Representation364Factory()
    board_to_input_converter: BoardToInput = Representation364BTI(
        representation_factory=representation_factory
    )
    return NNBoardEvaluator(
        net=net,
        output_and_value_converter=output_and_value_converter,
        board_to_input_converter=board_to_input_converter
    )
