import chipiron.players.move_selector.treevalue.trees as trees
import chipiron.players.move_selector.treevalue.nodes as nodes
from .index_data import NodeExplorationData, RecurZipfQuoolExplorationData
from .types import IndexComputationType
from .node_exploration_manager import UpdateIndexZipfFactoredProba, UpdateIndexGlobalMinChange, \
    UpdateIndexLocalMinChange, NodeExplorationIndexManager, NullNodeExplorationIndexManager


def create_exploration_index_manager(
        index_computation: IndexComputationType | None = None
) -> NodeExplorationIndexManager:
    if index_computation is None:
        node_exploration_manager: NodeExplorationIndexManager = NullNodeExplorationIndexManager()
    else:

        node_exploration_manager: NodeExplorationIndexManager
        match index_computation:
            case IndexComputationType.MinGlobalChange:
                node_exploration_manager = UpdateIndexGlobalMinChange()
            case IndexComputationType.RecurZipf:
                node_exploration_manager = UpdateIndexZipfFactoredProba()
            case IndexComputationType.MinLocalChange:
                node_exploration_manager = UpdateIndexLocalMinChange
            case other:
                raise ValueError(f'player creator: can not find {other} in {__name__}')

    return node_exploration_manager


def create_exploration_index_data(
        tree_node: nodes.TreeNode,
        index_computation: IndexComputationType | None = None
) -> NodeExplorationData | None:
    exploration_index_data: NodeExplorationData | None
    match index_computation:
        case None:
            exploration_index_data = None
        case IndexComputationType.MinLocalChange | IndexComputationType.MinGlobalChange:
            exploration_index_data = NodeExplorationData(tree_node=tree_node)
            if tree_node.is_root_node():
                exploration_index_data.index = 0
        case IndexComputationType.RecurZipf:
            exploration_index_data = RecurZipfQuoolExplorationData(tree_node=tree_node)
            if tree_node.is_root_node():
                exploration_index_data.index = 0
                exploration_index_data.zipf_factored_proba = 1

        case other:
            raise ValueError(f'not finding good case for {other} in file {__name__}')

    return exploration_index_data
