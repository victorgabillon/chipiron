from typing import Protocol, Callable
import chipiron.players.move_selector.treevalue.node_selector as node_selectors
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningType, OpeningInstructor
from dataclasses import dataclass
import random
from functools import partial
import chipiron.players.move_selector.treevalue.indices.node_indices as node_indices
from chipiron.players.move_selector.treevalue.indices.node_indices.factory import create_exploration_index_data
import chipiron.players.move_selector.treevalue.nodes as nodes
from chipiron.players.move_selector.treevalue.node_selector.sequool.factory import SequoolArgs
from chipiron.players.move_selector.treevalue.updates.index_updater import IndexUpdater

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
    ) -> node_indices.NodeExplorationData | None:
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

    node_selector_args: node_selectors.AllNodeSelectorArgs | None
    opening_type: OpeningType | None
    random_generator: random.Random | None
    index_computation: node_indices.IndexComputationType | None
    depth_index: bool = False

    def __post_init__(self):
        if isinstance(self.node_selector_args, SequoolArgs):
            a: SequoolArgs = self.node_selector_args
            self.depth_index: bool = a.recursive_selection_on_all_nodes
        else:
            self.depth_index: bool = False

    def create_node_selector_factory(
            self
    ) -> NodeSelectorFactory:
        # creates the opening instructor
        opening_instructor: OpeningInstructor | None = OpeningInstructor(
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

        index_updater: IndexUpdater | None
        if self.depth_index:
            index_updater = IndexUpdater()
        else:
            index_updater = None
        return index_updater

    def node_index_create(
            self,
            tree_node: nodes.TreeNode
    ) -> node_indices.NodeExplorationData | None:

        exploration_index_data: node_indices.NodeExplorationData | None = create_exploration_index_data(
            tree_node=tree_node,
            index_computation=self.index_computation,
            depth_index=self.depth_index
        )

        return exploration_index_data
