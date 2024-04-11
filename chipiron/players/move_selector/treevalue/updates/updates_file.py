"""
This module contains classes for managing update instructions in a batch.

Classes:
- UpdateInstructions: Represents update instructions for a single node.
- UpdateInstructionsBatch: Represents a batch of update instructions for multiple nodes.
"""

from dataclasses import dataclass
from typing import Self, Iterator

from chipiron.players.move_selector.treevalue.nodes import ITreeNode
from chipiron.utils.dict_of_numbered_dict_with_pointer_on_max import DictOfNumberedDictWithPointerOnMax
from .index_block import IndexUpdateInstructionsBlock
from .value_block import ValueUpdateInstructionsBlock


@dataclass(slots=True)
class UpdateInstructions:
    """
    Represents update instructions for a single node.

    Attributes:
    - value_block: The value update instructions block.
    - index_block: The index update instructions block.
    """

    value_block: ValueUpdateInstructionsBlock | None = None
    index_block: IndexUpdateInstructionsBlock | None = None

    def merge(
            self,
            an_update_instruction: Self,
            another_update_instruction: Self
    ) -> None:
        """
        Merges two update instructions into one.

        Args:
        - an_update_instruction: The first update instruction to merge.
        - another_update_instruction: The second update instruction to merge.
        """
        # Merge value blocks
        if an_update_instruction.value_block:
            if another_update_instruction.value_block:
                self.value_block = ValueUpdateInstructionsBlock()
                self.value_block.merge(
                    an_update_instruction.value_block,
                    another_update_instruction.value_block
                )
            else:
                self.value_block = an_update_instruction.value_block
        else:
            self.value_block = another_update_instruction.value_block

        # Merge index blocks
        if an_update_instruction.index_block:
            if another_update_instruction.index_block:
                self.index_block = IndexUpdateInstructionsBlock()
                self.index_block.merge(
                    an_update_instruction.index_block,
                    another_update_instruction.index_block
                )
            else:
                self.index_block = an_update_instruction.index_block
        else:
            self.index_block = another_update_instruction.index_block

    def print_info(self) -> None:
        """
        Prints information about the update instructions.
        """
        print('printing info of update instructions')
        assert (self.index_block is not None and self.value_block is not None)
        self.value_block.print_info()
        self.index_block.print_info()

    def empty(self) -> bool:
        """
        Checks if the update instructions are empty.

        Returns:
        - True if the update instructions are empty, False otherwise.
        """
        assert (self.value_block is not None)
        return self.value_block.empty() and (self.index_block is None or self.index_block.empty())


class UpdateInstructionsBatch:
    """
    Represents a batch of update instructions for multiple nodes.

    Attributes:
    - batch: The dictionary of update instructions for each node.
    """

    batch: DictOfNumberedDictWithPointerOnMax[ITreeNode, UpdateInstructions]

    def __init__(
            self,
            dictionary: dict[ITreeNode, UpdateInstructions] | None = None
    ) -> None:
        """
        Initializes the UpdateInstructionsBatch.

        Args:
        - dictionary: The initial dictionary of update instructions.
        """
        # batch is a dictionary of all the node from which a backward update should be started
        # it is a SortedDict where the keys involve the depth as the main sorting argument
        # this permits to easily give priority of update to the nodes with higher depth.
        # it should be less time-consuming because and less a redundant update depth per depth from the back
        # self.batch = MySortedDict()
        if dictionary is None:
            dictionary = {}
        self.batch = DictOfNumberedDictWithPointerOnMax()
        for node in dictionary:
            self.batch[node] = dictionary[node]

    def __setitem__(
            self,
            node: ITreeNode,
            value: UpdateInstructions
    ) -> None:
        """
        Sets the update instructions for a node.

        Args:
        - node: The node to set the update instructions for.
        - value: The update instructions for the node.
        """
        self.batch[node] = value

    def __getitem__(
            self,
            key: ITreeNode
    ) -> UpdateInstructions:
        """
        Gets the update instructions for a node.

        Args:
        - key: The node to get the update instructions for.

        Returns:
        - The update instructions for the node.
        """
        return self.batch[key]

    def __contains__(
            self,
            node: ITreeNode
    ) -> bool:
        """
        Checks if the batch contains update instructions for a node.

        Args:
        - node: The node to check.

        Returns:
        - True if the batch contains update instructions for the node, False otherwise.
        """
        return node in self.batch

    def __iter__(self) -> Iterator[ITreeNode]:
        """
        Returns an iterator over the nodes in the batch.

        Returns:
        - An iterator over the nodes in the batch.
        """
        raise Exception(f'fail in {__name__}')

    def popitem(self) -> tuple[ITreeNode, UpdateInstructions]:
        """
        Removes and returns the last node and its update instructions from the batch.

        Returns:
        - A tuple containing the node and its update instructions.
        """
        node, value = self.batch.popitem()
        return node, value

    def __bool__(self) -> bool:
        """
        Checks if the batch is empty.

        Returns:
        - True if the batch is not empty, False otherwise.
        """
        return bool(self.batch)

    def print_info(self) -> None:
        """
        Prints information about the update instructions batch.
        """
        print('UpdateInstructionsBatch: batch contains')
        raise Exception(f'not implemented in {__name__}')

    def merge(
            self,
            update_instructions_batch: Self
    ) -> None:
        """
        Merges another update instructions batch into this batch.

        Args:
        - update_instructions_batch: The update instructions batch to merge.
        """
        if update_instructions_batch is not None:
            for half_move in update_instructions_batch.batch.half_moves:
                for node in update_instructions_batch.batch.half_moves[half_move]:
                    if half_move in self.batch.half_moves and node in self.batch.half_moves[half_move]:
                        new_update_information = UpdateInstructions()
                        new_update_information.merge(
                            self.batch.half_moves[half_move][node],
                            update_instructions_batch.batch.half_moves[half_move][node]
                        )
                        self.batch[node] = new_update_information
                    else:
                        self.batch[node] = update_instructions_batch[node]
