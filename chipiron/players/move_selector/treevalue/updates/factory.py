from .algorithm_node_updater import AlgorithmNodeUpdater
from .minmax_evaluation_updater import MinMaxEvaluationUpdater
from .index_updater import IndexUpdater


def create_algorithm_node_updater(
        index_updater: IndexUpdater
) -> AlgorithmNodeUpdater:
    minmax_evaluation_updater: MinMaxEvaluationUpdater = MinMaxEvaluationUpdater()

    algorithm_node_updater: AlgorithmNodeUpdater = AlgorithmNodeUpdater(
        minmax_evaluation_updater=minmax_evaluation_updater,
        index_updater=index_updater
    )

    return algorithm_node_updater
