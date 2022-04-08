from abc import ABC, abstractmethod


class TreeNodeFactory(ABC):

    @abstractmethod
    def create_tree_node(self, board, half_move, count, father_node):
        pass

    @abstractmethod
    def update_after_node_creation(self, node, parent_node):
        pass
