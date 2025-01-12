"""
This module defines the Descendants and RangedDescendants classes.

Descendants:
- Represents a collection of descendants of a tree node at different half moves.
- Provides methods to add, remove, and access descendants.
- Keeps track of the number of descendants and the number of descendants at each half move.

RangedDescendants:
- Inherits from Descendants and adds the ability to track a range of half moves.
- Provides methods to check if a half move is within the current range or acceptable range.
- Allows adding and removing descendants within the range.
- Provides a method to get the range of half moves.

Note: The Descendants and RangedDescendants classes are used in the chipiron project for move selection in a game.
"""

import typing
from typing import Any, Iterator

from sortedcollections import ValueSortedDict

import chipiron.environments.chess.board as boards
from chipiron.environments import HalfMove
from chipiron.players.move_selector.treevalue.nodes import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.tree_traversal import (
    get_descendants,
)


class Descendants:
    """
    Represents a collection of descendants for a specific half move in a tree.

    Attributes:
        descendants_at_half_move (dict[HalfMove, dict[str, ITreeNode]]): A dictionary that maps a half move to a dictionary of descendants.
        number_of_descendants (int): The total number of descendants in the collection.
        number_of_descendants_at_half_move (dict[HalfMove, int]): A dictionary that maps a half move to the number of descendants at that half move.
        min_half_move (int | None): The minimum half move in the collection, or None if the collection is empty.
        max_half_move (int | None): The maximum half move in the collection, or None if the collection is empty.
    """

    descendants_at_half_move: dict[HalfMove, dict[boards.boardKey, ITreeNode[Any]]]
    number_of_descendants: int
    number_of_descendants_at_half_move: dict[HalfMove, int]
    min_half_move: int | None
    max_half_move: int | None

    def __init__(self) -> None:
        """
        Initializes a Descendants object.

        This method initializes the Descendants object by setting up the necessary attributes.

        Attributes:
        - descendants_at_half_move (dict): A dictionary to store the descendants at each half move.
        - number_of_descendants (int): The total number of descendants.
        - number_of_descendants_at_half_move (dict): A dictionary to store the number of descendants at each half move.
        - min_half_move (int or None): The minimum half move.
        - max_half_move (int or None): The maximum half move.
        """

        self.descendants_at_half_move = {}
        self.number_of_descendants = 0
        self.number_of_descendants_at_half_move = {}
        self.min_half_move = None
        self.max_half_move = None

    def iter_on_all_nodes(
        self,
    ) -> Iterator[tuple[HalfMove, boards.boardKey, ITreeNode[Any]]]:
        return (
            (hm, board_key, node)
            for hm, nodes_at_hm in self.descendants_at_half_move.items()
            for board_key, node in nodes_at_hm.items()
        )

    def keys(self) -> typing.KeysView[HalfMove]:
        """
        Returns a view of the keys in the descendants_at_half_move dictionary.

        Returns:
            typing.KeysView[HalfMove]: A view of the keys in the descendants_at_half_move dictionary.
        """
        return self.descendants_at_half_move.keys()

    def __setitem__(
        self, half_move: HalfMove, value: dict[boards.boardKey, ITreeNode[Any]]
    ) -> None:
        """
        Sets the descendants at a specific half move.

        Args:
            half_move (HalfMove): The half move at which to set the descendants.
            value (dict[str, ITreeNode]): The descendants to set.

        Returns:
            None
        """
        self.descendants_at_half_move[half_move] = value

    def __getitem__(self, half_move: HalfMove) -> dict[boards.boardKey, ITreeNode[Any]]:
        """
        Retrieve the descendants at a specific half move.

        Args:
            half_move (HalfMove): The half move to retrieve the descendants for.

        Returns:
            dict[str, ITreeNode]: A dictionary of descendants at the specified half move.
        """
        return self.descendants_at_half_move[half_move]

    def __iter__(self) -> typing.Iterator[HalfMove]:
        """
        Returns an iterator over the descendants at each half move.

        Returns:
            An iterator over the descendants at each half move.
        """
        return iter(self.descendants_at_half_move)

    def get_count(self) -> int:
        """
        Returns the number of descendants for the current node.

        Returns:
            int: The number of descendants.
        """
        return self.number_of_descendants

    def contains_node(self, node: ITreeNode[Any]) -> bool:
        """
        Checks if the descendants contain a specific node.

        Args:
            node (ITreeNode): The node to check for.

        Returns:
            bool: True if the descendants contain the node, False otherwise.
        """
        if (
            node.half_move in self.descendants_at_half_move
            and node.fast_rep in self[node.half_move]
        ):
            return True
        else:
            return False

    def remove_descendant(self, node: ITreeNode[Any]) -> None:
        """
        Removes a descendant node from the tree.

        Args:
            node (ITreeNode): The node to be removed.

        Returns:
            None
        """
        half_move = node.half_move
        fen = node.fast_rep

        self.number_of_descendants -= 1
        self[half_move].pop(fen)
        self.number_of_descendants_at_half_move[half_move] -= 1
        if self.number_of_descendants_at_half_move[half_move] == 0:
            self.number_of_descendants_at_half_move.pop(half_move)
            self.descendants_at_half_move.pop(half_move)

    def empty(self) -> bool:
        """
        Check if the descendants are empty.

        Returns:
            bool: True if the number of descendants is 0, False otherwise.
        """
        return self.number_of_descendants == 0

    def add_descendant(self, node: ITreeNode[Any]) -> None:
        """
        Adds a descendant node to the tree.

        Args:
            node (ITreeNode): The descendant node to be added.

        Returns:
            None
        """
        half_move: HalfMove = node.half_move
        board_key: boards.boardKey = node.fast_rep

        if half_move in self.descendants_at_half_move:
            assert board_key not in self.descendants_at_half_move[half_move]
            self.descendants_at_half_move[half_move][board_key] = node
            self.number_of_descendants_at_half_move[half_move] += 1
        else:
            self.descendants_at_half_move[half_move] = {board_key: node}
            self.number_of_descendants_at_half_move[half_move] = 1
        self.number_of_descendants += 1

    def __len__(self) -> int:
        """
        Returns the number of descendants at the current half move.

        :return: The number of descendants at the current half move.
        :rtype: int
        """
        return len(self.descendants_at_half_move)

    def print_info(self) -> None:
        """
        Prints information about the descendants.

        This method prints the number of descendants and their corresponding half moves.
        It also prints the ID and fast representation of each descendant.

        Returns:
            None
        """
        print("---here are the ", self.get_count(), " descendants.")
        for half_move in self:
            print(
                "half_move: ",
                half_move,
                "| (",
                self.number_of_descendants_at_half_move[half_move],
                "descendants)",
            )  # ,                  end='| ')
            for descendant in self[half_move].values():
                print(descendant.id, descendant.fast_rep, end=" ")
            print("")

    def print_stats(self) -> None:
        """
        Prints the statistics of the descendants.

        This method prints the number of descendants at each half move.

        Returns:
            None
        """
        print("---here are the ", self.get_count(), " descendants")
        for half_move in self:
            print(
                "half_move: ",
                half_move,
                "| (",
                self.number_of_descendants_at_half_move[half_move],
                "descendants)",
            )

    def test(self) -> None:
        """
        This method performs a series of assertions to validate the descendants data structure.
        It checks if the number of descendants at each half move matches the number of descendants stored.
        It also checks if the sum of the lengths of all descendants at each half move matches the total number of descendants.
        """
        assert set(self.descendants_at_half_move.keys()) == set(
            self.number_of_descendants_at_half_move
        )
        sum_ = 0
        for half_move in self:
            sum_ += len(self[half_move])
        assert self.number_of_descendants == sum_

        for half_move in self:
            assert self.number_of_descendants_at_half_move[half_move] == len(
                self[half_move]
            )

    def test_2(self, root_node: ITreeNode[Any]) -> None:
        """
        Test the descendants of a given root node.

        Args:
            root_node (ITreeNode): The root node to test.

        Returns:
            None
        """
        all_descendants = get_descendants(root_node)

        # self.print_info()
        for d in all_descendants:
            if d.half_move not in self.descendants_at_half_move:
                assert d.half_move in self.descendants_at_half_move
            if d.fast_rep not in self.descendants_at_half_move[d.half_move]:
                assert d.fast_rep in self.descendants_at_half_move[d.half_move]

        for half_move in self.descendants_at_half_move:
            for d in self[half_move].values():
                assert d in all_descendants


class RangedDescendants(Descendants):
    """
    Represents a collection of descendants with a range of half moves.

    Attributes:
        min_half_move (int | None): The minimum half move in the range.
        max_half_move (int | None): The maximum half move in the range.
    """

    min_half_move: int | None
    max_half_move: int | None

    def __init__(self) -> None:
        """
        Initializes a Descendants object.
        """
        super().__init__()
        self.min_half_move = None
        self.max_half_move = None

    def __str__(self) -> str:
        """
        Returns a string representation of the Descendants object.

        The string includes information about each half move and its descendants.

        Returns:
            str: A string representation of the Descendants object.
        """
        string: str = ""
        for half_move in self:
            string += f"half_move: {half_move} | ({self.number_of_descendants_at_half_move[half_move]} descendants)\n"
            for descendant in self[half_move].values():
                string += f"{descendant.id} "
            string += "\n"
        return string

    def is_new_generation(self, half_move: HalfMove) -> bool:
        """
        Checks if the given half move is a new generation.

        Args:
            half_move (HalfMove): The half move to check.

        Returns:
            bool: True if the half move is a new generation, False otherwise.
        """
        if self.min_half_move is not None:
            assert self.max_half_move is not None
            return half_move == self.max_half_move + 1
        else:
            return True

    def is_in_the_current_range(self, half_move: int) -> bool:
        """
        Checks if the given half_move is within the current range.

        Args:
            half_move (int): The half_move to check.

        Returns:
            bool: True if the half_move is within the range, False otherwise.
        """

        if self.min_half_move is not None and self.max_half_move is not None:
            return self.max_half_move >= half_move >= self.min_half_move
        else:
            return False

    def is_in_the_acceptable_range(self, half_move: int) -> bool:
        """
        Checks if the given half_move is within the acceptable range.

        Args:
            half_move (int): The half_move to check.

        Returns:
            bool: True if the half_move is within the acceptable range, False otherwise.
        """
        if self.min_half_move is not None and self.max_half_move is not None:
            return self.max_half_move + 1 >= half_move >= self.min_half_move
        else:
            return True

    def add_descendant(self, node: ITreeNode[Any]) -> None:
        """
        Adds a descendant node to the tree.

        Args:
            node (ITreeNode): The descendant node to be added.

        Returns:
            None
        """
        half_move: int = node.half_move
        board_key: boards.boardKey = node.fast_rep

        assert self.is_in_the_acceptable_range(half_move)
        if self.is_in_the_current_range(half_move):
            if half_move in self.descendants_at_half_move:
                assert board_key not in self.descendants_at_half_move[half_move]
            self.descendants_at_half_move[half_move][board_key] = node
            self.number_of_descendants_at_half_move[half_move] += 1
        else:  # half_move == len(self.descendants_at_half_move)
            assert self.is_new_generation(half_move)
            self.descendants_at_half_move[half_move] = {board_key: node}
            self.number_of_descendants_at_half_move[half_move] = 1
            if self.max_half_move is not None:
                self.max_half_move += 1
            else:
                self.min_half_move = half_move
                self.max_half_move = half_move
        self.number_of_descendants += 1

    def remove_descendant(self, node: ITreeNode[Any]) -> None:
        """
        Removes a descendant node from the tree.

        Args:
            node (ITreeNode): The node to be removed.

        Returns:
            None
        """
        half_move: int = node.half_move
        fen = node.fast_rep

        self.number_of_descendants -= 1
        self[half_move].pop(fen)
        self.number_of_descendants_at_half_move[half_move] -= 1
        if self.number_of_descendants_at_half_move[half_move] == 0:
            self.number_of_descendants_at_half_move.pop(half_move)
            self.descendants_at_half_move.pop(half_move)
            if half_move == self.max_half_move:
                self.max_half_move -= 1
            if half_move == self.min_half_move:
                self.min_half_move += 1
            assert self.max_half_move is not None
            assert self.min_half_move is not None
            if self.max_half_move < self.min_half_move:
                self.max_half_move = None
                self.min_half_move = None
                assert self.number_of_descendants == 0

    def range(self) -> range:
        """
        Returns a range object representing the half moves range.

        The range starts from the minimum half move and ends at the maximum half move.

        Returns:
            range: A range object representing the half moves range.
        """

        assert self.max_half_move is not None
        assert self.min_half_move is not None
        return range(self.min_half_move, self.max_half_move + 1)

    #  def update(
    #          self,
    #          new_descendants: typing.Self
    #  ) -> RangedDescendants:
    #      really_new_descendants : RangedDescendants()#

    #        for half_move in new_descendants.range():
    #            if half_move in self:
    #               really_new_descendants_keys = set(new_descendants[half_move].keys()).difference(
    #                  set(self[half_move].keys()))
    #          else:
    #              really_new_descendants_keys = set(new_descendants[half_move].keys())
    #          for key in really_new_descendants_keys:
    #              really_new_descendants.add_descendant(new_descendants[half_move][key])
    #             self.add_descendant(new_descendants[half_move][key])

    #   # really_new_descendants.print_info()

    #    return really_new_descendants

    def merge(self, descendant_1: typing.Self, descendant_2: typing.Self) -> None:
        """
        Merges the descendants of two nodes into the current node.

        Args:
            descendant_1 (typing.Self): The first descendant node.
            descendant_2 (typing.Self): The second descendant node.

        Returns:
            None
        """
        half_moves_range = set(descendant_1.keys()) | set(descendant_2.keys())
        assert len(half_moves_range) > 0
        self.min_half_move = min(half_moves_range)
        self.max_half_move = max(half_moves_range)
        for half_move in half_moves_range:
            if descendant_1.is_in_the_current_range(half_move):
                if descendant_2.is_in_the_current_range(half_move):
                    #  print('dd',type(self.descendants_at_half_move),type())
                    # in python 3.9 we can use a |
                    self.descendants_at_half_move[half_move] = {
                        **descendant_1[half_move],
                        **descendant_2[half_move],
                    }
                    self.number_of_descendants_at_half_move[half_move] = len(
                        self[half_move]
                    )
                    assert self.number_of_descendants_at_half_move[half_move] == len(
                        {**descendant_1[half_move], **descendant_2[half_move]}
                    )
                else:
                    self.descendants_at_half_move[half_move] = descendant_1[half_move]
                    self.number_of_descendants_at_half_move[half_move] = (
                        descendant_1.number_of_descendants_at_half_move[half_move]
                    )
            else:
                self.descendants_at_half_move[half_move] = descendant_2[half_move]
                self.number_of_descendants_at_half_move[half_move] = (
                    descendant_2.number_of_descendants_at_half_move[half_move]
                )
            self.number_of_descendants += self.number_of_descendants_at_half_move[
                half_move
            ]

    def test(self) -> None:
        """
        Perform a test on the descendants object.

        This method checks the validity of the descendants object by asserting various conditions.
        If the `min_half_move` attribute is None, it asserts that `max_half_move` is also None and
        `number_of_descendants` is 0.
        Otherwise, it asserts that `max_half_move` and `min_half_move` are not None, and checks if all half moves
        between `min_half_move` and `max_half_move` are present in `descendants_at_half_move` dictionary.
        Finally, it iterates over all half moves in the descendants object and asserts that each half move
        is within the current range.

        Returns:
            None
        """
        super().test()
        if self.min_half_move is None:
            assert self.max_half_move is None
            assert self.number_of_descendants == 0
        else:
            assert self.max_half_move is not None
            assert self.min_half_move is not None
            for i in range(self.min_half_move, self.max_half_move + 1):
                assert i in self.descendants_at_half_move.keys()
        for half_move in self:
            assert self.is_in_the_current_range(half_move)

    def print_info(self) -> None:
        """
        Prints information about the descendants.

        This method calls the `print_info` method of the parent class and then prints the count of descendants,
        the minimum half move, and the maximum half move.

        Returns:
            None
        """
        super().print_info()
        print(
            "---here are the ",
            self.get_count(),
            " descendants. min:",
            self.min_half_move,
            ". max:",
            self.max_half_move,
        )


class SortedDescendants(Descendants):
    # todo is there a difference between sorted descendant nd sorted value descendant? below?

    """
    Represents a class that stores sorted descendants of a tree node at different half moves.
    Inherits from the Descendants class.
    """

    sorted_descendants_at_half_move: dict[int, dict[ITreeNode[Any], float]]

    def __init__(self) -> None:
        super().__init__()
        self.sorted_descendants_at_half_move = {}

    def update_value(self, node: ITreeNode[Any], value: float) -> None:
        """
        Updates the value of a descendant node.

        Args:
            node (ITreeNode): The descendant node.
            value (float): The new value for the descendant node.
        """
        self.sorted_descendants_at_half_move[node.half_move][node] = value

    def add_descendant_with_val(self, node: ITreeNode[Any], value: float) -> None:
        """
        Adds a descendant node with its corresponding value.

        Args:
            node (ITreeNode): The descendant node to add.
            value (float): The value of the descendant node.
        """
        super().add_descendant(node)
        half_move = node.half_move

        if half_move in self.sorted_descendants_at_half_move:
            assert node not in self.sorted_descendants_at_half_move[half_move]
            self.sorted_descendants_at_half_move[half_move][node] = value
        else:  # half_move == len(self.descendants_at_half_move)
            self.sorted_descendants_at_half_move[half_move] = {node: value}

        assert self.contains_node(node)

    def test(self) -> None:
        """
        Performs a test to ensure the integrity of the data structure.
        """
        super().test()
        assert len(self.sorted_descendants_at_half_move) == len(
            self.descendants_at_half_move
        )
        assert (
            self.sorted_descendants_at_half_move.keys()
            == self.descendants_at_half_move.keys()
        )
        for half_move in self.sorted_descendants_at_half_move:
            assert len(self.sorted_descendants_at_half_move[half_move]) == len(
                self.descendants_at_half_move[half_move]
            )

    def print_info(self) -> None:
        """
        Prints information about the sorted descendants.
        """
        super().print_info()
        print("sorted")
        for half_move in self:
            print(
                "half_move: ",
                half_move,
                "| (",
                self.number_of_descendants_at_half_move[half_move],
                "descendants)",
            )  # ,                  end='| ')
            for descendant, value in self.sorted_descendants_at_half_move[
                half_move
            ].items():
                print(
                    descendant.id, descendant.fast_rep, "(" + str(value) + ")", end=" "
                )
            print("")

    def remove_descendant(self, node: ITreeNode[Any]) -> None:
        """
        Removes a descendant node from the data structure.

        Args:
            node (ITreeNode): The descendant node to remove.
        """
        super().remove_descendant(node)
        half_move = node.half_move
        self.sorted_descendants_at_half_move[half_move].pop(node)
        if half_move not in self.number_of_descendants_at_half_move:
            self.sorted_descendants_at_half_move.pop(half_move)
        assert not self.contains_node(node)

    def contains_node(self, node: ITreeNode[Any]) -> bool:
        """
        Checks if a descendant node is present in the data structure.

        Args:
            node (ITreeNode): The descendant node to check.

        Returns:
            bool: True if the descendant node is present, False otherwise.
        """
        reply_base = super().contains_node(node)
        if (
            node.half_move in self.descendants_at_half_move
            and node in self.sorted_descendants_at_half_move[node.half_move]
        ):
            rep = True
        else:
            rep = False
        assert reply_base == rep
        return rep


class SortedValueDescendants(Descendants):
    """
    Represents a class for managing sorted descendants with associated values.
    Inherits from the `Descendants` class.
    """

    sorted_descendants_at_half_move: dict[typing.Any, typing.Any]

    def __init__(self) -> None:
        """
        Initializes a Sorted Value Descendants object.
        """
        super().__init__()
        self.sorted_descendants_at_half_move = {}

    def update_value(self, node: ITreeNode[Any], value: float) -> None:
        """
        Updates the value associated with a given node.

        Args:
            node (ITreeNode): The node to update the value for.
            value (float): The new value to associate with the node.

        Returns:
            None
        """
        self.sorted_descendants_at_half_move[node.half_move][node] = value

    def add_descendant_val(self, node: ITreeNode[Any], value: float) -> None:
        """
        Adds a descendant node with an associated value.

        Args:
            node (ITreeNode): The descendant node to add.
            value (float): The value associated with the descendant node.

        Returns:
            None
        """
        super().add_descendant(node)
        half_move = node.half_move

        if half_move in self.sorted_descendants_at_half_move:
            assert node not in self.sorted_descendants_at_half_move[half_move]
            self.sorted_descendants_at_half_move[half_move][node] = value
        else:
            self.sorted_descendants_at_half_move[half_move] = ValueSortedDict(
                {node: value}
            )

    def test(self) -> None:
        """
        Performs a test to ensure the integrity of the sorted descendants.

        Returns:
            None
        """
        super().test()
        assert len(self.sorted_descendants_at_half_move) == len(
            self.descendants_at_half_move
        )
        assert (
            self.sorted_descendants_at_half_move.keys()
            == self.descendants_at_half_move.keys()
        )
        for half_move in self.sorted_descendants_at_half_move:
            assert len(self.sorted_descendants_at_half_move[half_move]) == len(
                self.descendants_at_half_move[half_move]
            )

    def print_info(self) -> None:
        """
        Prints information about the sorted descendants.

        Returns:
            None
        """
        super().print_info()
        print("sorted")
        for half_move in self:
            print(
                "half_move: ",
                half_move,
                "| (",
                self.number_of_descendants_at_half_move[half_move],
                "descendants)",
            )
            for descendant, value in self.sorted_descendants_at_half_move[
                half_move
            ].items():
                print(str(descendant.id) + "(" + str(value) + ")", end=" ")
            print("")

    def remove_descendant(self, node: ITreeNode[Any]) -> None:
        """
        Removes a descendant node.

        Args:
            node (ITreeNode): The descendant node to remove.

        Returns:
            None
        """
        super().remove_descendant(node)
        half_move = node.half_move

        self.sorted_descendants_at_half_move[half_move].pop(node)
        if half_move not in self.number_of_descendants_at_half_move:
            self.sorted_descendants_at_half_move.pop(half_move)
