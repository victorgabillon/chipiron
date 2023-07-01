from .algorithm_node_tree_manager import AlgorithmNodeTreeManager, TreeManager
from chipiron.players.treevalue import node_factory
import chipiron.players.treevalue.updates as upda
from .node_evaluators_wrapper import  NodeEvaluatorsWrapper, EvaluationQueries


def create_algorithm_node_tree_manager(
        node_evaluators_wrapper: NodeEvaluatorsWrapper,
        algorithm_node_factory: node_factory.AlgorithmNodeFactory) -> AlgorithmNodeTreeManager:
    tree_manager: TreeManager = TreeManager(
        node_factory=algorithm_node_factory)

    algorithm_node_updater: upda.AlgorithmNodeUpdater = upda.create_algorithm_node_updater()

    evaluation_queries: EvaluationQueries = EvaluationQueries()

    algorithm_node_tree_manager: AlgorithmNodeTreeManager = AlgorithmNodeTreeManager(
        node_evaluators_wrapper=node_evaluators_wrapper,
        tree_manager=tree_manager,
        algorithm_node_updater=algorithm_node_updater,
        evaluation_queries=evaluation_queries
    )

    return algorithm_node_tree_manager
