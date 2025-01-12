"""
This module contains the implementation of the SearchFactory class, which is an abstract factory that creates
dependent factories for selecting nodes to open, creating indices, and updating indices. These factories need to
operate on the same data, so they are created in a coherent way.

The SearchFactory class provides methods to create the node selector factory, the index updater, and to create
node indices for a given tree node.

Classes:
- SearchFactory: An abstract factory for creating dependent factories for selecting nodes to open, creating indices,
and updating indices.

Protocols:
- SearchFactoryP: A protocol that defines the abstract methods for the SearchFactory class.

Functions:
- create_node_selector_factory: A function that creates the node selector factory.
- create_node_index_updater: A function that creates the index updater.
- node_index_create: A function that creates node indices for a given tree node.
"""

import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Protocol

import chipiron.players.move_selector.treevalue.indices.node_indices as node_indices
import chipiron.players.move_selector.treevalue.node_selector as node_selectors
import chipiron.players.move_selector.treevalue.nodes as nodes
from chipiron.players.move_selector.treevalue.indices.node_indices.factory import (
    create_exploration_index_data,
)
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import (
    OpeningInstructor,
    OpeningType,
)
from chipiron.players.move_selector.treevalue.node_selector.sequool.factory import (
    SequoolArgs,
)
from chipiron.players.move_selector.treevalue.updates.index_updater import IndexUpdater

NodeSelectorFactory = Callable[[], node_selectors.NodeSelector]


class SearchFactoryP(Protocol):
    """
    The abstract Factory that creates the following dependent factories in charge of selecting nodes to open
    - the node selector
    - the index creator
    - the index updater
    These three classes needs to operate on the same data, so they need to be created in a coherent way
    """

    def create_node_selector_factory(self) -> NodeSelectorFactory:
        """
        Creates a NodeSelectorFactory object.

        Returns:
            NodeSelectorFactory: The created NodeSelectorFactory object.
        """
        ...

    def create_node_index_updater(self) -> IndexUpdater | None:
        """
        Creates and returns an instance of the IndexUpdater class, which is responsible for updating the node index.

        Returns:
            IndexUpdater | None: An instance of the IndexUpdater class if successful, None otherwise.
        """
        ...

    def node_index_create(
        self, tree_node: nodes.TreeNode[Any]
    ) -> node_indices.NodeExplorationData | None:
        """
        Creates a node index for the given tree node.

        Args:
            tree_node (nodes.TreeNode): The tree node for which to create the index.

        Returns:
            node_indices.NodeExplorationData | None: The created node index, or None if the index could not be created.
        """
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

    def __post_init__(self) -> None:
        """
        Initializes the object after it has been created.

        This method is automatically called after the object has been initialized.
        It sets the value of `depth_index` based on the type of `node_selector_args`.

        If `node_selector_args` is an instance of `SequoolArgs`, then `depth_index`
        is set to the value of `recursive_selection_on_all_nodes` attribute of `node_selector_args`.
        Otherwise, `depth_index` is set to False.
        """
        if isinstance(self.node_selector_args, SequoolArgs):
            a: SequoolArgs = self.node_selector_args
            self.depth_index: bool = a.recursive_selection_on_all_nodes
        else:
            self.depth_index: bool = False

    def create_node_selector_factory(self) -> NodeSelectorFactory:
        """
        Creates the node selector factory .

        Returns:
            A callable object that creates the node selector.

        Raises:
            AssertionError: If the random generator is not provided.
        """
        # creates the opening instructor

        assert self.random_generator is not None
        opening_instructor: OpeningInstructor | None = (
            OpeningInstructor(
                opening_type=self.opening_type, random_generator=self.random_generator
            )
            if self.opening_type is not None
            else None
        )

        assert self.node_selector_args is not None
        assert opening_instructor is not None

        node_selector_create: NodeSelectorFactory = partial(
            node_selectors.create,
            args=self.node_selector_args,
            opening_instructor=opening_instructor,
            random_generator=self.random_generator,
        )
        return node_selector_create

    def create_node_index_updater(self) -> IndexUpdater | None:
        """
        Creates the index updater.

        Returns:
            An instance of the IndexUpdater class if depth indexing is enabled, otherwise None.
        """
        index_updater: IndexUpdater | None
        if self.depth_index:
            index_updater = IndexUpdater()
        else:
            index_updater = None
        return index_updater

    def node_index_create(
        self, tree_node: nodes.TreeNode[Any]
    ) -> node_indices.NodeExplorationData | None:
        """
        Creates node indices for a given tree node.

        Args:
            tree_node: The tree node for which to create the node indices.

        Returns:
            An instance of the NodeExplorationData class if depth indexing is enabled, otherwise None.
        """
        exploration_index_data: node_indices.NodeExplorationData | None = (
            create_exploration_index_data(
                tree_node=tree_node,
                index_computation=self.index_computation,
                depth_index=self.depth_index,
            )
        )

        return exploration_index_data
