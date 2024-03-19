from .algorithm_node_updater import AlgorithmNodeUpdater
from .index_updater import IndexUpdater
from .minmax_evaluation_updater import MinMaxEvaluationUpdater


def create_algorithm_node_updater(
        index_updater: IndexUpdater | None
) -> AlgorithmNodeUpdater:
    minmax_evaluation_updater: MinMaxEvaluationUpdater = MinMaxEvaluationUpdater()

    algorithm_node_updater: AlgorithmNodeUpdater = AlgorithmNodeUpdater(
        minmax_evaluation_updater=minmax_evaluation_updater,
        index_updater=index_updater
    )

    return algorithm_node_updater
