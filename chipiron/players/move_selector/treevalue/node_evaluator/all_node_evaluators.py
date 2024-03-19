from enum import Enum


class NodeEvaluatorTypes(str, Enum):
    NeuralNetwork: str = 'neural_network'
    BasicEvaluation: str = 'basic_evaluation'
