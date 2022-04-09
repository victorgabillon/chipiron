from abc import ABC, abstractmethod


class TreeNodeFactory(ABC):

    @abstractmethod
    def create(self,
               board,
               half_move,
               count,
               parent_node,
               board_depth):
        pass

    @abstractmethod
    def update_after_node_creation(self, node, parent_node):
        pass
