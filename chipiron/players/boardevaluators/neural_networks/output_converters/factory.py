from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    PointOfView,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)
from chipiron.players.boardevaluators.neural_networks.output_converters.output_value_converter import (
    IdentityConverter,
    OutputValueConverter,
    PlayerToMoveValueToValueWhiteConverter,
)


def create_output_converter(model_output_type: ModelOutputType) -> OutputValueConverter:
    output_value_converter: OutputValueConverter
    point_of_view: PointOfView = model_output_type.point_of_view
    match point_of_view:
        case PointOfView.WHITE:
            output_value_converter = IdentityConverter()
        case PointOfView.PLAYER_TO_MOVE:
            output_value_converter = PlayerToMoveValueToValueWhiteConverter()
        case other:
            raise Exception(f"Not a valid output converter: {other} in file{__name__}")

    return output_value_converter
