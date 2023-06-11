from .algorithm_node_tree_manager import AlgorithmNodeTreeManager, TreeManager
from chipiron.players.treevalue import node_factory
import chipiron.players.treevalue.updates as upda


def create_algorithm_node_tree_manager(
        board_evaluators_wrapper,
        algorithm_node_factory: node_factory.AlgorithmNodeFactory) -> AlgorithmNodeTreeManager:
    tree_manager: TreeManager = TreeManager(
        node_factory=algorithm_node_factory)

    algorithm_node_updater: upda.AlgorithmNodeUpdater = upda.create_algorithm_node_updater()


    algorithm_node_tree_manager: AlgorithmNodeTreeManager
    algorithm_node_tree_manager = AlgorithmNodeTreeManager(board_evaluators_wrapper=board_evaluators_wrapper,
                                                           tree_manager=tree_manager,
                                                           algorithm_node_updater=algorithm_node_updater)

    return algorithm_node_tree_manager
