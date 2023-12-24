from .algorithm_node_tree_manager import AlgorithmNodeTreeManager, TreeManager
from players.move_selector.treevalue import node_factory
import players.move_selector.treevalue.updates as upda
from players.move_selector.treevalue.node_evaluator import NodeEvaluator, EvaluationQueries


def create_algorithm_node_tree_manager(
        node_evaluator: NodeEvaluator,
        algorithm_node_factory: node_factory.AlgorithmNodeFactory) -> AlgorithmNodeTreeManager:
    tree_manager: TreeManager = TreeManager(
        node_factory=algorithm_node_factory)

    algorithm_node_updater: upda.AlgorithmNodeUpdater = upda.create_algorithm_node_updater()

    evaluation_queries: EvaluationQueries = EvaluationQueries()

    algorithm_node_tree_manager: AlgorithmNodeTreeManager = AlgorithmNodeTreeManager(
        node_evaluator=node_evaluator,
        tree_manager=tree_manager,
        algorithm_node_updater=algorithm_node_updater,
        evaluation_queries=evaluation_queries
    )

    return algorithm_node_tree_manager
