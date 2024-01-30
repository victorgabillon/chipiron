from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode
from chipiron.utils.small_tools import Interval
from dataclasses import dataclass, field


@dataclass(slots=True)
class NodeExplorationData:
    tree_node: TreeNode
    index: float | None = None

    def dot_description(self):
        return f'index:{self.index}'


@dataclass(slots=True)
class RecurZipfQuoolExplorationData(NodeExplorationData):
    # the 'proba' associated by recursively multiplying 1/rank of the node with the max zipf_factor of the parents
    zipf_factored_proba: float | None = None

    def dot_description(self):
        return f'index:{self.index} zipf_factored_proba:{self.zipf_factored_proba}'


@dataclass(slots=True)
class MinMaxPathValue(NodeExplorationData):
    min_path_value: float | None = None
    max_path_value: float | None = None

    def dot_description(self):
        return f'min_path_value: {self.min_path_value}, max_path_value: {self.max_path_value}'


@dataclass(slots=True)
class IntervalExplo(NodeExplorationData):
    interval: Interval = field(default_factory=Interval)

    def dot_description(self):
        return f'min_interval_value: {self.min_path_value}, max_interval_value: {self.max_path_value}'
