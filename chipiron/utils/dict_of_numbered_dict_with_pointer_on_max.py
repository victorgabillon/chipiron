"""
Module for DictOfNumberedDictWithPointerOnMax class.
"""

from typing import Protocol


class HasHalfMove(Protocol):
    @property
    def half_move(self) -> int:
        """
        Get the half move count of the node.

        Returns:
            The half move count of the node.
        """


class DictOfNumberedDictWithPointerOnMax[T_Key: HasHalfMove, T_Value]:
    """
    A dictionary-like data structure that stores numbered dictionaries and keeps track of the maximum half move.

    Attributes:
        half_moves (dict[int, dict[T_Key, T_Value]]): A dictionary that stores numbered dictionaries.
        max_half_move (int | None): The maximum half move value.

    Methods:
        __setitem__(self, node: T_Key, value: T_Value) -> None: Adds an item to the data structure.
        __getitem__(self, node: T_Key) -> T_Value: Retrieves an item from the data structure.
        __bool__(self) -> bool: Checks if the data structure is non-empty.
        __contains__(self, node: T_Key) -> bool: Checks if an item is present in the data structure.
        popitem(self) -> tuple[T_Key, T_Value]: Removes and returns the item with the maximum half move value.
    """

    def __init__(self) -> None:
        self.half_moves: dict[int, dict[T_Key, T_Value]] = {}
        self.max_half_move: int | None = None

    def __setitem__(self, node: T_Key, value: T_Value) -> None:
        """
        Adds an item to the data structure.

        Args:
            node (T_Key): The key of the item.
            value (T_Value): The value of the item.

        Returns:
            None
        """
        half_move = node.half_move
        if self.max_half_move is None:
            self.max_half_move = half_move
        else:
            self.max_half_move = max(half_move, self.max_half_move)
        if half_move in self.half_moves:
            self.half_moves[half_move][node] = value
        else:
            self.half_moves[half_move] = {node: value}

        assert self.max_half_move == max(self.half_moves)

    def __getitem__(self, node: T_Key) -> T_Value:
        """
        Retrieves an item from the data structure.

        Args:
            node (T_Key): The key of the item.

        Returns:
            T_Value: The value of the item.

        Raises:
            KeyError: If the item is not found in the data structure.
        """
        return self.half_moves[node.half_move][node]

    def __bool__(self) -> bool:
        """
        Checks if the data structure is non-empty.

        Returns:
            bool: True if the data structure is non-empty, False otherwise.
        """
        return bool(self.half_moves)

    def __contains__(self, node: T_Key) -> bool:
        """
        Checks if an item is present in the data structure.

        Args:
            node (T_Key): The key of the item.

        Returns:
            bool: True if the item is present, False otherwise.
        """
        if node.half_move not in self.half_moves:
            return False
        else:
            return node in self.half_moves[node.half_move]

    def popitem(self) -> tuple[T_Key, T_Value]:
        """
        Removes and returns the item with the maximum half move value.

        Returns:
            tuple[T_Key, T_Value]: The key-value pair of the removed item.

        Raises:
            AssertionError: If the data structure is empty.
        """
        assert self.max_half_move is not None
        popped: tuple[T_Key, T_Value] = self.half_moves[self.max_half_move].popitem()
        if not self.half_moves[self.max_half_move]:
            del self.half_moves[self.max_half_move]
            if self.half_moves:
                self.max_half_move = max(self.half_moves.keys())
            else:
                self.max_half_move = None

        return popped

    # def sort_dic(self):
    #    self.dic = dict(sorted(self.dic.items(), key=lambda item: item[0]))
    # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
