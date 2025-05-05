from typing import TypeAlias

from chipiron.players.boardevaluators.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from chipiron.players.boardevaluators.neural_networks.models.tranformer_one import (
    TransformerArgs,
)

NNModelTypeArgs: TypeAlias = MultiLayerPerceptronArgs | TransformerArgs
