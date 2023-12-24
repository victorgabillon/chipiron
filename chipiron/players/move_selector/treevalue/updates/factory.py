from .algorithm_node_updater import AlgorithmNodeUpdater
from .minmax_evaluation_updater import MinMaxEvaluationUpdater


def create_algorithm_node_updater() -> AlgorithmNodeUpdater:
    minmax_evaluation_updater: MinMaxEvaluationUpdater = MinMaxEvaluationUpdater()

    algorithm_node_updater: AlgorithmNodeUpdater = AlgorithmNodeUpdater(
        minmax_evaluation_updater=minmax_evaluation_updater)
    return algorithm_node_updater
