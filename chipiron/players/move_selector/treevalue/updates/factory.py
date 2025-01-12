"""
This module provides a factory function to create an instance of AlgorithmNodeUpdater.

The AlgorithmNodeUpdater is responsible for updating the algorithm node in a tree structure.

The factory function `create_algorithm_node_updater` takes an optional `index_updater` parameter and returns an instance of AlgorithmNodeUpdater.

Example usage:
    algorithm_node_updater = create_algorithm_node_updater(index_updater)
"""

from .algorithm_node_updater import AlgorithmNodeUpdater
from .index_updater import IndexUpdater
from .minmax_evaluation_updater import MinMaxEvaluationUpdater


def create_algorithm_node_updater(
    index_updater: IndexUpdater | None,
) -> AlgorithmNodeUpdater:
    """
    Creates an instance of AlgorithmNodeUpdater.

    Args:
        index_updater (IndexUpdater | None): The index updater object.

    Returns:
        AlgorithmNodeUpdater: An instance of AlgorithmNodeUpdater.

    """
    minmax_evaluation_updater: MinMaxEvaluationUpdater = MinMaxEvaluationUpdater()

    algorithm_node_updater: AlgorithmNodeUpdater = AlgorithmNodeUpdater(
        minmax_evaluation_updater=minmax_evaluation_updater, index_updater=index_updater
    )

    return algorithm_node_updater
