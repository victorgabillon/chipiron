from dataclasses import dataclass, field

from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode
from chipiron.utils.small_tools import Interval


@dataclass
class NodeExplorationData:
    tree_node: TreeNode
    index: float | None = None

    def dot_description(self):
        return f'index:{self.index}'


@dataclass
class RecurZipfQuoolExplorationData(NodeExplorationData):
    # the 'proba' associated by recursively multiplying 1/rank of the node with the max zipf_factor of the parents
    zipf_factored_proba: float | None = None

    def dot_description(self):
        return f'index:{self.index} zipf_factored_proba:{self.zipf_factored_proba}'


@dataclass
class MinMaxPathValue(NodeExplorationData):
    min_path_value: float | None = None
    max_path_value: float | None = None

    def dot_description(self):
        return f'min_path_value: {self.min_path_value}, max_path_value: {self.max_path_value}'


@dataclass
class IntervalExplo(NodeExplorationData):
    interval: Interval | None = field(default_factory=Interval)

    def dot_description(self):
        if self.interval is None:
            return 'None'
        else:
            return f'min_interval_value: {self.interval.min_value}, max_interval_value: {self.interval.max_value}'


@dataclass
class MaxDepthDescendants(NodeExplorationData):
    max_depth_descendants: int = 0

    def update_from_child(
            self,
            child_max_depth_descendants: int
    ) -> bool:
        previous_index = self.max_depth_descendants
        new_index: int = max(
            self.max_depth_descendants,
            child_max_depth_descendants + 1
        )
        self.max_depth_descendants = new_index
        has_index_changed: bool = new_index != previous_index

        return has_index_changed

    def dot_description(self):
        return f'max_depth_descendants: {self.max_depth_descendants}'
