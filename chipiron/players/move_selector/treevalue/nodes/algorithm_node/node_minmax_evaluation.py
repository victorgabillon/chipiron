"""
This module contains the implementation of the NodeMinmaxEvaluation class, which represents a node in a tree structure
 used for the Minimax algorithm evaluation.

The NodeMinmaxEvaluation class stores information about the evaluation of a tree node, including the estimated value
 for the white player, the computed value using the Minimax procedure, the best node sequence, and the children of
  the tree node sorted by their evaluations.

It also provides methods for accessing and manipulating the evaluation values, determining the subjective value from
 the point of view of the player to move, finding the best child node, checking if the node is over, and printing
 information about the node.

Note: This code snippet is a partial implementation and may require additional code to work properly.
"""

# todo maybe further split values from over?

import math
import typing
from dataclasses import dataclass, field
from random import choice
from typing import Any, Protocol, Self

import chess

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    BoardEvaluation,
    FloatyBoardEvaluation,
    ForcedOutcome,
)
from chipiron.players.boardevaluators.over_event import OverEvent
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.my_value_sorted_dict import sort_dic
from chipiron.utils.small_tools import nth_key


@typing.runtime_checkable
# Class created to avoid circular import and defines what is seen and needed by the NodeMinmaxEvaluation class
class NodeWithValue(ITreeNode[Any], Protocol):
    """
    Represents a node with a value in a tree structure.

    Attributes:
        minmax_evaluation (NodeMinmaxEvaluation): The minmax evaluation associated with the node.
        tree_node (TreeNode): The tree node associated with the node.
    """

    minmax_evaluation: "NodeMinmaxEvaluation"
    tree_node: TreeNode[Any]


@dataclass(slots=True)
class NodeMinmaxEvaluation:
    r"""
    Represents a node in a tree structure used for the Minimax algorithm evaluation.

    Attributes:
        tree_node (TreeNode): A reference to the original tree node that is evaluated.
        value_white_evaluator (float | None): The absolute value with respect to the white player as estimated
        by an evaluator.
        value_white_minmax (float | None): The absolute value with respect to the white player as computed from
        the value_white_* of the descendants of this node (self) by a Minimax procedure.
        best_node_sequence (list[ITreeNode]): The sequence of best nodes found during the Minimax evaluation.
        children_sorted_by_value\_ (dict[ITreeNode, Any]): The children of the tree node kept in a dictionary
        that can be sorted by their evaluations.
        best_index_for_value (int): The index of the best value in the children_sorted_by_value dictionary.
        children_not_over (list[ITreeNode]): The list of children that have not yet been found to be over.
        over_event (OverEvent): The event that determines if the node is over.
    """

    # a reference to the original tree node that is evaluated
    tree_node: TreeNode[NodeWithValue]

    # absolute value wrt to white player as estimated by an evaluator
    value_white_evaluator: float | None = None

    # absolute value wrt to white player as computed from the value_white_* of the descendants
    # of this node (self) by a minmax procedure.
    value_white_minmax: float | None = None

    # the sequence of best moves from this node
    best_move_sequence: list[moveKey] = field(default_factory=list)

    # the children of the tree node are kept in a dictionary that can be sorted by their evaluations ()

    # children_sorted_by_value records subjective values of children by descending order
    # subjective value means the values is from the point of view of player_to_move
    # careful, I have hard coded in the self.best_child() function the descending order for
    # fast access to the best element, so please do not change!
    # self.children_sorted_by_value_vsd = ValueSortedDict({})
    moves_sorted_by_value_: dict[moveKey, Any] = field(default_factory=dict)

    # self.children_sorted_by_value = {}

    # convention of descending order, careful if changing read above!!
    best_index_for_value: int = 0

    # the list of moves that have not yet be found to be over
    # using atm a list instead of set as atm python set are not insertion ordered which adds randomness
    # and makes debug harder
    moves_not_over: list[moveKey] = field(default_factory=list)

    # creating a base Over event that is set to None
    over_event: OverEvent = field(default_factory=OverEvent)

    @property
    def moves_sorted_by_value(self) -> dict[moveKey, Any]:
        """
        Returns a dictionary containing the moves of the node sorted by their values.

        Returns:
            dict[IMove, Any]: A dictionary where the keys are the moves in the node and the values
            are their corresponding values.
        """
        return self.moves_sorted_by_value_

    def get_value_white(self) -> float:
        """Returns the best estimation of the value for white in this node.

        Returns:
            float: The best estimation of the value for white in this node.
        """
        assert self.value_white_minmax is not None
        return self.value_white_minmax

    def set_evaluation(self, evaluation: float) -> None:
        """sets the evaluation from the board evaluator

        Args:
            evaluation (float): The evaluation value to be set.

        Returns:
            None
        """
        self.value_white_evaluator = evaluation
        self.value_white_minmax = (
            evaluation  # base value before knowing values of the children
        )

    def subjective_value_(self, value_white: float) -> float:
        """
        Return the subjective value of `value_white` from the point of view of the `self.tree_node.player_to_move`.

        The subjective value is calculated based on the player to move. If the player to move is `chess.WHITE`, then the
        `value_white` is returned as is. Otherwise, the negative of `value_white` is returned.

        Args:
            value_white (float): The value from the point of view of the white player.

        Returns:
            float: The subjective value of `value_white` based on the player to move.
        """
        subjective_value = (
            value_white
            if self.tree_node.player_to_move == chess.WHITE
            else -value_white
        )
        return subjective_value

    def subjective_value(self) -> float:
        """Return the subjective value of self.value_white from the point of view of the self.tree_node.player_to_move.

        If the player to move is chess.WHITE, the subjective value is self.get_value_white().
        If the player to move is not chess.WHITE, the subjective value is -self.get_value_white().

        Returns:
            float: The subjective value of self.value_white.
        """
        subjective_value = (
            self.get_value_white()
            if self.tree_node.player_to_move == chess.WHITE
            else -self.get_value_white()
        )
        return subjective_value

    def subjective_value_of(self, another_node_eval: Self) -> float:
        """
        Calculates the subjective value of the current node evaluation based on the player to move.

        Args:
            another_node_eval (Self): The evaluation of another node.

        Returns:
            float: The subjective value of the current node evaluation.
        """
        if self.tree_node.player_to_move == chess.WHITE:
            subjective_value = another_node_eval.get_value_white()
        else:
            subjective_value = -another_node_eval.get_value_white()
        return subjective_value

    def best_move(self) -> moveKey | None:
        """
        Returns the best move node based on the subjective value.

        Returns:
            The best move based on the subjective value, or None if there are no move open.
        """
        best_move: moveKey | None
        if self.moves_sorted_by_value:
            best_move = next(iter(self.moves_sorted_by_value))
        else:
            best_move = None
        return best_move

    def best_move_not_over(self) -> moveKey:
        """
        Returns the best move that is not leading to a game-over.

        Returns:
            The best move that is not leading to a game-over.

        Raises:
            Exception: If no move is found that is not over.
        """
        move: moveKey
        for move in self.moves_sorted_by_value:
            child = self.tree_node.moves_children[move]
            assert child is not None
            if not child.is_over():
                return move
        raise Exception("Not ok")

    def best_move_value(self) -> float | None:
        """
        Returns the value of the best move.

        If the `moves_sorted_by_value` dictionary is not empty, it returns the value of the first child node with
        the highest subjective value. Otherwise, it returns None.

        Returns:
            float | None: The value of the best child node, or None if there are no child nodes.
        """
        best_value: float | None
        # fast way to access first key with the highest subjective value
        if self.moves_sorted_by_value:
            best_value = next(iter(self.moves_sorted_by_value.values()))
        else:
            best_value = None
        return best_value

    def second_best_move(self) -> moveKey:
        """
        Returns the second-best move based on the subjective value.

        Returns:
            The second-best move.
        """
        assert len(self.moves_sorted_by_value) >= 2
        # fast way to access second key with the highest subjective value
        second_best_move: moveKey = nth_key(self.moves_sorted_by_value, 1)
        return second_best_move

    def is_over(self) -> bool:
        """
        Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.over_event.is_over()

    def is_win(self) -> bool:
        """
        Checks if the current game state is a win.

        Returns:
            bool: True if the game state is a win, False otherwise.
        """
        return self.over_event.is_win()

    def is_draw(self) -> bool:
        """
        Checks if the current game state is a draw.

        Returns:
            bool: True if the game state is a draw, False otherwise.
        """
        return self.over_event.is_draw()

    def is_winner(self, player: chess.Color) -> bool:
        """
        Determines if the specified player is the winner.

        Args:
            player (chess.Color): The color of the player to check.

        Returns:
            bool: True if the player is the winner, False otherwise.
        """
        return self.over_event.is_winner(player)

    def print_moves_sorted_by_value(self) -> None:
        """
        Prints the moves sorted by their subjective sort value.

        The method iterates over the move_sorted_by_value dictionary and prints each move along with its
        subjective sort value. The output is formatted as follows:
        "<move>: <subjective_sort_value> $$"

        Returns:
            None
        """
        print(
            "here are the ", len(self.moves_sorted_by_value), " move sorted by value: "
        )
        move: moveKey
        for move, subjective_sort_value in self.moves_sorted_by_value.items():
            print(
                self.tree_node.board.get_uci_from_move_key(move),
                subjective_sort_value[0],
                end=" $$ ",
            )
        print("")

    def print_moves_sorted_by_value_and_exploration(self) -> None:
        """
        Prints the move of the node sorted by their value and exploration.

        This method prints the moves of the node along with their subjective sort value.
        The moves are sorted based on their value and exploration.

        Args:
            None

        Returns:
            None
        """
        move: moveKey
        chipiron_logger.info(
            f"here are the {len(self.moves_sorted_by_value)} moves sorted by value: "
        )
        string_info: str = ""
        for move, subjective_sort_value in self.moves_sorted_by_value.items():
            string_info += f" {self.tree_node.board.get_uci_from_move_key(move)} {subjective_sort_value[0]} $$ "
        chipiron_logger.info(string_info)

    def print_moves_not_over(self) -> None:
        """
        Prints the moves that are not over.

        This method prints the moves that are not marked as 'over'.
        It iterates over the `moves_not_over` list and prints each child's ID.

        Returns:
            None
        """
        print("here are the ", len(self.moves_not_over), " move not over: ", end=" ")
        for move in self.moves_not_over:
            print(move, end=" ")
        print(" ")

    def print_info(self) -> None:
        """
        Prints information about the node.

        This method prints the ID of the node, the moves of its children, the children sorted by value,
        and the children that are not over.
        """
        print("Soy el Node", self.tree_node.id)
        self.tree_node.print_moves_children()
        self.print_moves_sorted_by_value()
        self.print_moves_not_over()
        # todo probably more to print...

    def record_sort_value_of_child(self, move: moveKey) -> None:
        """Stores the subjective value of the move in the self.moves_sorted_by_value (automatically sorted).

        Args:
            move (IMove): The move whose value needs to be recorded.

        Returns:
            None
        """
        # - moves_sorted_by_value records subjective value of moves by descending order
        # therefore we have to convert the value_white of children into a subjective value that depends
        # on the player to move
        # - subjective best move/children is at index 0 however sortedValueDict are sorted ascending (best index: -1),
        # therefore for white we have negative values
        child = self.tree_node.moves_children[move]
        assert child is not None
        child_value_white = child.minmax_evaluation.get_value_white()
        subjective_value_of_child = (
            -child_value_white
            if self.tree_node.player_to_move == chess.WHITE
            else child_value_white
        )
        if self.is_over():
            # the shorter the check the better now
            self.moves_sorted_by_value_[move] = (
                subjective_value_of_child,
                -len(child.minmax_evaluation.best_move_sequence),
                child.tree_node.id,
            )

        else:
            # the longer the line the better now
            self.moves_sorted_by_value_[move] = (
                subjective_value_of_child,
                len(child.minmax_evaluation.best_move_sequence),
                child.tree_node.id,
            )

    def are_equal_values[T](self, value_1: T, value_2: T) -> bool:
        """
        Check if two values are equal.

        Args:
            value_1 (T): The first value to compare.
            value_2 (T): The second value to compare.

        Returns:
            bool: True if the values are equal, False otherwise.
        """
        return value_1 == value_2

    def are_considered_equal_values[T](
        self, value_1: tuple[T], value_2: tuple[T]
    ) -> bool:
        """
        Check if two values are considered equal.

        Args:
            value_1 (tuple[T]): The first value to compare.
            value_2 (tuple[T]): The second value to compare.

        Returns:
            bool: True if the values are considered equal, False otherwise.
        """
        return value_1[:2] == value_2[:2]

    def are_almost_equal_values(self, value_1: float, value_2: float) -> bool:
        """
        Check if two float values are almost equal within a small epsilon.

        Args:
            value_1 (float): The first value to compare.
            value_2 (float): The second value to compare.

        Returns:
            bool: True if the values are almost equal, False otherwise.
        """
        epsilon = 0.01
        return value_1 > value_2 - epsilon and value_2 > value_1 - epsilon

    def becoming_over_from_children(self) -> None:
        """This node is asked to switch to over status.

        This method is called when the node is requested to switch to the "over" status. It performs the necessary
        operations to update the node's status and determine the winner.

        Raises:
            AssertionError: If the node is already in the "over" status.

        """
        assert not self.is_over()

        # becoming over triggers a full update record_sort_value_of_child
        # where ties are now broken to reach over as fast as possible
        # todo we should reach it asap if we are winning and think about what to ddo in other scenarios....
        move: moveKey
        for move in self.tree_node.moves_children:
            self.record_sort_value_of_child(move=move)

        # fast way to access first key with the highest subjective value
        best_move: moveKey | None = self.best_move()
        assert best_move is not None
        best_child = self.tree_node.moves_children[best_move]
        assert best_child is not None
        self.over_event.becomes_over(
            how_over=best_child.minmax_evaluation.over_event.how_over,
            who_is_winner=best_child.minmax_evaluation.over_event.who_is_winner,
            termination=best_child.minmax_evaluation.over_event.termination,
        )

    def update_over(self, moves_with_updated_over: set[moveKey]) -> bool:
        """
        Update the over_event of the node based on notification of change of over_event in children.

        Args:
            moves_with_updated_over (set[IMove]): A set of chess moves linking to the children
                nodes that have been updated with their over_event.

        Returns:
            bool: True if the node has become newly over, False otherwise.
        """

        is_newly_over = False

        # Two cases can make this node (self) become over:
        # 1. One of the children of this node is over and is a win for the node.player_to_move.
        # 2. All children are now over, then choose your best over event (choose draw if you can avoid a loss).

        for move in moves_with_updated_over:
            child = self.tree_node.moves_children[move]
            assert child is not None
            assert child.is_over()
            if move in self.moves_not_over:
                self.moves_not_over.remove(move)

            # Check if child is already not in children_not_over.
            if not self.is_over() and child.minmax_evaluation.is_winner(
                self.tree_node.player_to_move
            ):
                self.becoming_over_from_children()
                is_newly_over = True

        # Check if all children are over but not winning for self.tree_node.player_to_move.
        if not self.is_over() and not self.moves_not_over:
            self.becoming_over_from_children()
            is_newly_over = True

        return is_newly_over

    def update_moves_values(self, moves_to_consider: set[moveKey]) -> None:
        """
        Updates the values of the moves based on the given set of moves to consider.

        Args:
            moves_to_consider (set[IMove]): The set of moves to consider.

        Returns:
            None
        """
        for move in moves_to_consider:
            self.record_sort_value_of_child(move=move)
        self.moves_sorted_by_value_ = sort_dic(self.moves_sorted_by_value_)

    def sort_moves_not_over(self) -> list[moveKey]:
        """
        Sorts the moves that are not over based on their value.

        Returns:
            A sorted list of moves that are not over.
        """
        # todo: looks like the determinism of the sort induces some determinisin the play like always
        #  playing the same actions when a lot of them have equal value: introduce some randomness?
        return [
            move for move in self.moves_sorted_by_value if move in self.moves_not_over
        ]  # todo is this a fast way to do it?

    def update_value_minmax(self) -> None:
        """
        Updates the minmax value for the current node based on the best child node's evaluation.

        If all the children of the current node have been evaluated, the minmax value is set to the best child's
        evaluation value. Otherwise, if not all children have been evaluated, the minmax value is determined by
        comparing the best child's evaluation value with the current node's own evaluation value.

        Note: The evaluation values are specific to the white player.

        Returns:
            None
        """
        best_move: moveKey | None = self.best_move()
        assert best_move is not None
        best_child = self.tree_node.moves_children[best_move]
        assert best_child is not None
        if self.tree_node.all_legal_moves_generated:
            self.value_white_minmax = best_child.minmax_evaluation.get_value_white()
        elif self.tree_node.player_to_move == chess.WHITE:
            assert self.value_white_evaluator is not None
            self.value_white_minmax = max(
                best_child.minmax_evaluation.get_value_white(),
                self.value_white_evaluator,
            )
        else:
            assert self.value_white_evaluator is not None
            self.value_white_minmax = min(
                best_child.minmax_evaluation.get_value_white(),
                self.value_white_evaluator,
            )

    def update_best_move_sequence(
        self, moves_with_updated_best_move_seq: set[moveKey]
    ) -> bool:
        """Updates the best move sequence based on the notification from children nodes identified through their
        corresponding move.

        Args:
            moves_with_updated_best_move_seq (set[IMove]): A set of move that have
                notified an updated best-move sequence.

        Returns:
            bool: True if self.best_move_sequence is modified, False otherwise.
        """
        has_best_move_seq_changed: bool = False
        best_move: moveKey = self.best_move_sequence[0]
        best_node: NodeWithValue | None = self.tree_node.moves_children[best_move]

        if best_move in moves_with_updated_best_move_seq and best_node is not None:
            self.best_move_sequence = [
                best_move
            ] + best_node.minmax_evaluation.best_move_sequence
            has_best_move_seq_changed = True

        return has_best_move_seq_changed

    def one_of_best_children_becomes_best_next_node(self) -> None:
        """Triggered when the value of the current best move does not match the best value.

        This method selects one of the best children nodes as the next best node based on a specific condition.
        It updates the `best_move_sequence` attribute with the selected child node and its corresponding best move sequence.

        Raises:
            AssertionError: If the number of best children is not equal to 1 when `how_equal_` is set to 'equal'.
            AssertionError: If the selected best child is not an instance of `NodeWithValue`.
            AssertionError: If the `best_node_sequence` attribute is empty after updating.

        """
        how_equal_: str = "equal"
        best_moves: list[moveKey] = self.get_all_of_the_best_moves(how_equal=how_equal_)
        if how_equal_ == "equal":
            assert len(best_moves) == 1
        best_move = choice(best_moves)
        best_child = self.tree_node.moves_children[best_move]
        # best_child = best_children[len(best_children) - 1]  # for debug!!
        assert best_child is not None
        self.best_move_sequence = [
            best_move
        ] + best_child.minmax_evaluation.best_move_sequence
        assert self.best_move_sequence

    def is_value_subjectively_better_than_evaluation(self, value_white: float) -> bool:
        """
        Checks if the given value_white is subjectively better than the value_white_evaluator.

        Args:
            value_white (float): The value to compare with the value_white_evaluator.

        Returns:
            bool: True if the value_white is subjectively better than the value_white_evaluator, False otherwise.
        """
        subjective_value = self.subjective_value_(value_white)
        assert self.value_white_evaluator is not None
        return subjective_value >= self.value_white_evaluator

    def minmax_value_update_from_children(
        self, moves_with_updated_value: set[moveKey]
    ) -> tuple[bool, bool]:
        """
        Updates the value and best move of the node based on the updated values of its children.

        Args:
            moves_with_updated_value (set[IMove]): A set of moves with updated values.

        Returns:
            tuple[bool, bool]: A tuple containing two boolean values indicating whether the value and best move have
            changed.
        """

        # todo to be tested!!

        # updates value
        value_white_before_update = self.get_value_white()

        best_move_before_update: moveKey | None = self.best_move()
        self.update_moves_values(moves_to_consider=moves_with_updated_value)
        self.update_value_minmax()

        value_white_after_update = self.get_value_white()
        has_value_changed: bool = value_white_before_update != value_white_after_update

        # # updates best_move #todo maybe split in two functions but be careful one has to be done oft the other
        if best_move_before_update is None:
            best_child_before_update_not_the_best_anymore = True
        else:
            # here we compare the values in the self.children_sorted_by_value which might include more
            # than just the basic values #todo make that more clear at some point maybe even creating a value object
            updated_value_of_best_child_before_update = self.moves_sorted_by_value[
                best_move_before_update
            ]
            best_value_children_after = self.best_move_value()
            best_child_before_update_not_the_best_anymore = not self.are_equal_values(
                updated_value_of_best_child_before_update, best_value_children_after
            )

        best_move_seq_before_update: list[moveKey] = self.best_move_sequence.copy()
        if self.tree_node.all_legal_moves_generated:
            if best_child_before_update_not_the_best_anymore:
                self.one_of_best_children_becomes_best_next_node()
        else:
            # we only consider a child as best if it is more promising than the evaluation of self
            # in self.value_white_evaluator
            assert best_move_before_update is not None
            best_child_before_update = self.tree_node.moves_children[
                best_move_before_update
            ]
            assert best_child_before_update is not None
            if self.is_value_subjectively_better_than_evaluation(
                best_child_before_update.minmax_evaluation.get_value_white()
            ):
                self.one_of_best_children_becomes_best_next_node()
            else:
                self.best_move_sequence = []
        best_move_seq_after_update = self.best_move_sequence
        has_best_node_seq_changed = (
            best_move_seq_before_update != best_move_seq_after_update
        )

        return has_value_changed, has_best_node_seq_changed

    def dot_description(self) -> str:
        """
        Returns a string representation of the node's description in DOT format.

        The description includes the values of `value_white_minmax` and `value_white_evaluator`,
        as well as the best move sequence and the over event tag.

        Returns:
            A string representation of the node's description in DOT format.
        """
        value_mm = (
            "{:.3f}".format(self.value_white_minmax)
            if self.value_white_minmax is not None
            else "None"
        )
        value_eval = (
            "{:.3f}".format(self.value_white_evaluator)
            if self.value_white_evaluator is not None
            else "None"
        )
        return (
            "\n wh_val_mm: "
            + value_mm
            + "\n wh_val_eval: "
            + value_eval
            + "\n moves*"
            + self.description_best_move_sequence()
            + "\nover: "
            + self.over_event.get_over_tag()
        )

    def description_best_move_sequence(self) -> str:
        """
        Returns a string representation of the best move sequence.

        This method iterates over the best node sequence and constructs a string representation
        of the moves in the sequence. Each move is appended to the result string, separated by an underscore.

        Returns:
            A string representation of the best move sequence.
        """
        res = ""
        move: moveKey
        for move in self.best_move_sequence:
            res += "_" + str(move)
        return res

    def description_tree_visualizer_move(self, child: Any) -> str:
        """
        Returns a string representation of the move for the tree visualizer.

        Parameters:
        - child (Any): The child node representing the move.

        Returns:
        - str: A string representation of the move for the tree visualizer.
        """
        return ""

    def test(self) -> None:
        """
        This method is used to test the functionality of the algorithm_node's test method.
        It calls various test methods to ensure the correctness of the algorithm_node's behavior.
        """
        self.tree_node.test()
        self.test_values()
        self.test_over()
        self.test_children_not_over()
        # self.test_best_node_sequence()

    def test_children_not_over(self) -> None:
        """
        Test if the children of the algorithm node are correctly classified as over or not over.

        This method iterates over the moves_children dictionary of the tree_node and checks if each child is correctly
        classified as over or not over. It asserts that if a child is over, it should not be present in the
        children_not_over set, and if a child is not over, it should be present in the children_not_over set.
        """
        child: NodeWithValue | None
        for move, child in self.tree_node.moves_children.items():
            assert child is not None
            if child.is_over():
                assert move not in self.moves_not_over
            else:
                assert move in self.moves_not_over

    def test_over(self) -> None:
        """to be coded again"""

        ...

    def test_values(self) -> None:
        """
        Test the values of the algorithm node and its children.

        This method asserts various conditions to ensure that the values of the algorithm node and its children are correct.
        It checks if the values of the children are consistent with the minmax evaluation values.
        It also checks if the maximum or minimum value among the children matches the value of the algorithm node,
        depending on the player to move.
        """
        value_children: list[float] = []
        for move, child in self.tree_node.moves_children.items():
            assert child is not None
            assert (
                self.moves_sorted_by_value[move][0]
                * (1 - 2 * self.tree_node.player_to_move)
                == child.minmax_evaluation.get_value_white()
            )
            value_children.append(child.minmax_evaluation.get_value_white())
        if self.tree_node.moves_children:
            if self.tree_node.player_to_move == chess.WHITE:
                assert max(value_children) == self.get_value_white()
            if self.tree_node.player_to_move == chess.BLACK:
                assert min(value_children) == self.get_value_white()
            for ind, move in enumerate(self.moves_sorted_by_value):
                child = self.tree_node.moves_children[move]
                assert child is not None
                if ind == 0:
                    assert (
                        child.minmax_evaluation.get_value_white()
                        == self.get_value_white()
                    )
                    before = child
                else:
                    before = child
                    if self.tree_node.player_to_move == chess.WHITE:
                        assert (
                            before.minmax_evaluation.get_value_white()
                            >= child.minmax_evaluation.get_value_white()
                        )
                    if self.tree_node.player_to_move == chess.BLACK:
                        assert (
                            before.minmax_evaluation.get_value_white()
                            <= child.minmax_evaluation.get_value_white()
                        )
        else:
            pass
            # todo test board value

    def print_best_line(self) -> None:
        """
        Prints the best line from the current node to the leaf node.

        The best line is determined by following the sequence of child nodes with the highest values.
        Each child node is printed along with its corresponding move and node ID.

        Returns:
            None
        """
        info_string: str = f"Best line from node {str(self.tree_node.id)}: "
        minmax: NodeMinmaxEvaluation = self
        for move in self.best_move_sequence:
            child = minmax.tree_node.moves_children[move]
            assert child is not None
            info_string += f"{move} ({str(child.tree_node.id)}) "
            minmax = child.minmax_evaluation
        chipiron_logger.info(info_string)

    def my_logit(self, x: float) -> float:
        """
        Applies the logit function to the input value.

        Args:
            x (float): The input value.

        Returns:
            float: The result of applying the logit function to the input value.
        """
        y = min(max(x, 0.000000000000000000000001), 0.9999999999999999)
        return math.log(y / (1 - y)) * max(
            1, abs(x)
        )  # the * min(1,x) is a hack to prioritize game over

    def get_all_of_the_best_moves(self, how_equal: str | None = None) -> list[moveKey]:
        """
        Returns a list of all the best moves based on the specified equality criteria.

        Args:
            how_equal (str | None): The equality criteria to determine the best moves.
                Possible values are 'equal', 'considered_equal', 'almost_equal', 'almost_equal_logistic'.
                Defaults to None.

        Returns:
            list[IMove]: A list of IMove representing the best moves.

        """
        best_moves: list[moveKey] = []
        best_move: moveKey | None = self.best_move()
        assert best_move is not None
        best_value = self.moves_sorted_by_value[best_move]
        move: moveKey
        for move in self.moves_sorted_by_value:
            if how_equal == "equal":
                if self.are_equal_values(self.moves_sorted_by_value[move], best_value):
                    best_moves.append(move)
                    assert len(best_moves) == 1
            elif how_equal == "considered_equal":
                if self.are_considered_equal_values(
                    self.moves_sorted_by_value[move], best_value
                ):
                    best_moves.append(move)
            elif how_equal == "almost_equal":
                if self.are_almost_equal_values(
                    self.moves_sorted_by_value[move][0], best_value[0]
                ):
                    best_moves.append(move)
            elif how_equal == "almost_equal_logistic":
                best_value_logit = self.my_logit(best_value[0] * 0.5 + 0.5)
                child_value_logit = self.my_logit(
                    self.moves_sorted_by_value[move][0] * 0.5 + 0.5
                )
                if self.are_almost_equal_values(child_value_logit, best_value_logit):
                    best_moves.append(move)
        return best_moves

    def evaluate(self) -> BoardEvaluation:
        if self.over_event.is_over():
            return ForcedOutcome(
                outcome=self.over_event, line=[move for move in self.best_move_sequence]
            )
        else:
            return FloatyBoardEvaluation(value_white=self.value_white_minmax)
