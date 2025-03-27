from dataclasses import dataclass

from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    PointOfView,
)


# this type is about the type of representation input fed to NN models
@dataclass
class ModelOutputType:
    point_of_view: PointOfView
