from chipiron.players.move_selector.treevalue.indices.index_manager.node_exploration_manager import \
    UpdateIndexZipfFactoredProba, \
    UpdateIndexGlobalMinChange, \
    UpdateIndexLocalMinChange, NodeExplorationIndexManager, NullNodeExplorationIndexManager
from chipiron.players.move_selector.treevalue.indices.node_indices.index_types import IndexComputationType


def create_exploration_index_manager(
        index_computation: IndexComputationType | None = None
) -> NodeExplorationIndexManager:
    node_exploration_manager: NodeExplorationIndexManager
    if index_computation is None:
        node_exploration_manager = NullNodeExplorationIndexManager()
    else:
        match index_computation:
            case IndexComputationType.MinGlobalChange:
                node_exploration_manager = UpdateIndexGlobalMinChange()
            case IndexComputationType.RecurZipf:
                node_exploration_manager = UpdateIndexZipfFactoredProba()
            case IndexComputationType.MinLocalChange:
                node_exploration_manager = UpdateIndexLocalMinChange()
            case other:
                raise ValueError(f'player creator: can not find {other} in {__name__}')

    return node_exploration_manager
