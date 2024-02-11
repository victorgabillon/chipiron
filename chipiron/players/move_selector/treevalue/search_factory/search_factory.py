from typing import Protocol, Callable
import chipiron.players.move_selector.treevalue.node_selector as node_selectors
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningType, OpeningInstructor
from dataclasses import dataclass
import random
from functools import partial
import chipiron.players.move_selector.treevalue.node_indices as node_indices
from chipiron.players.move_selector.treevalue.node_indices.factory import create_exploration_index_data
import chipiron.players.move_selector.treevalue.nodes as nodes
from chipiron.players.move_selector.treevalue.node_selector.sequool.factory import SequoolArgs

NodeSelectorFactory = Callable[[], node_selectors.NodeSelector]


# NodeIndexFactory = Callable[[], node_indices.NodeExplorationData]


class SearchFactoryP(Protocol):
    """
    The abstract Factory that creates the following dependent factories in charge of selecting nodes to open
    - the node selector
    - the index creator
    - the index updater
    These three classes needs to operate on the same data, so they need to be created in a coherent way
    """

    def create_node_selector_factory(
            self
    ) -> NodeSelectorFactory:
        ...

    def create_node_index_updater(self):
        ...

    def node_index_create(
            self,
            tree_node: nodes.TreeNode
    ) -> node_indices.NodeExplorationData:
        ...


@dataclass
class SearchFactory:
    """
    The abstract Factory that creates the following dependent factories in charge of selecting nodes to open
    - the node selector
    - the index creator
    - the index updater
    These three classes needs to operate on the same data, so they need to be created in a coherent way
    """

    node_selector_args: node_selectors.AllNodeSelectorArgs
    opening_type: OpeningType
    random_generator: random.Random
    index_computation: node_indices.IndexComputationType | None

    def create_node_selector_factory(
            self
    ) -> NodeSelectorFactory:
        # creates the opening instructor
        opening_instructor: OpeningInstructor = OpeningInstructor(
            self.opening_type, self.random_generator
        ) if self.opening_type is not None else None

        node_selector_create: NodeSelectorFactory = partial(
            node_selectors.create,
            args=self.node_selector_args,
            opening_instructor=opening_instructor,
            random_generator=self.random_generator,
        )
        return node_selector_create

    def create_node_index_updater(self):
        create_exploration_index_data(
            tree_node=tree_node,
            index_computation=self.index_computation
        )

    def node_index_create(
            self,
            tree_node: nodes.TreeNode
    ) -> node_indices.NodeExplorationData:
        if isinstance(self.node_selector_args, SequoolArgs):
            a: SequoolArgs = self.node_selector_args
            depth_index:bool = a.recursive_selection_on_all_nodes
        else:
            depth_index: bool = False
        exploration_index_data: node_indices.NodeExplorationData = create_exploration_index_data(
            tree_node=tree_node,
            index_computation=self.index_computation,
            depth_index=depth_index
        )

        return exploration_index_data
