"""
Defining the AlgorithmNodeTreeManager class
"""

import chess
import chipiron.players.move_selector.treevalue.nodes as node
import chipiron.players.move_selector.treevalue.trees as trees
import chipiron.players.move_selector.treevalue.node_selector as node_sel
import chipiron.players.move_selector.treevalue.updates as upda

from .tree_expander import TreeExpansion, TreeExpansions
from .tree_manager import TreeManager

from chipiron.players.move_selector.treevalue.node_evaluator import NodeEvaluator, EvaluationQueries


# todo should we use a discount? and discounted per round reward?
# todo maybe convenient to seperate this object into openner updater and dsiplayer
# todo have the reward with a discount
# DISCOUNT = 1/.99999


class AlgorithmNodeTreeManager:
    """
    This class that and manages a tree by opening new nodes and updating the values and indexes on the nodes.
    It wraps around the Tree Manager class as it has a tree_manager as member and adds functionality as this handles
    trees with nodes that are of the class AlgorithmNode (managing the value for instance)
    """

    tree_manager: TreeManager
    algorithm_node_updater: upda.AlgorithmNodeUpdater
    evaluation_queries: EvaluationQueries
    node_evaluator: NodeEvaluator

    def __init__(
            self,
            node_evaluator: NodeEvaluator,
            tree_manager: TreeManager,
            algorithm_node_updater: upda.AlgorithmNodeUpdater,
            evaluation_queries: EvaluationQueries
    ) -> None:
        """
        init the class and setting the members
        """
        self.node_evaluator = node_evaluator
        self.tree_manager = tree_manager
        self.algorithm_node_updater = algorithm_node_updater
        self.evaluation_queries = evaluation_queries

    def open_node_move(
            self,
            tree: trees.MoveAndValueTree,
            parent_node: node.AlgorithmNode,
            move: chess.Move
    ) -> TreeExpansion:
        """

        Args:
            tree: the tree to open
            parent_node: the node to open
            move: to move to open with

        Returns: the tree expansions

        """
        tree_expansion: TreeExpansion = self.tree_manager.open_node_move(tree=tree,
                                                                         parent_node=parent_node,
                                                                         move=move)
        parent_node.minmax_evaluation.children_not_over.append(
            tree_expansion.child_node)  # default action checks for over event are performed later

        return tree_expansion

    def open_instructions(
            self,
            tree: trees.MoveAndValueTree,
            opening_instructions: node_sel.OpeningInstructions
    ) -> TreeExpansions:
        """

        Args:
            tree: the tree object to open
            opening_instructions: the opening instructions

        Returns: the expansions that have been performed

        """

        # place to store the tree expansion logs generated by the openings
        tree_expansions: TreeExpansions = TreeExpansions()

        opening_instruction: node_sel.OpeningInstruction
        for opening_instruction in opening_instructions.values():
            # open
            tree_expansion: TreeExpansion = self.open_node_move(
                tree=tree,
                parent_node=opening_instruction.node_to_open,
                move=opening_instruction.move_to_play
            )

            # concatenate the tree expansions
            tree_expansions.add(tree_expansion=tree_expansion)

        tree_expansion: TreeExpansion
        for tree_expansion in tree_expansions.expansions_with_node_creation:
            # TODO give the tree expansion to the function directly

            self.node_evaluator.add_evaluation_query(node=tree_expansion.child_node,
                                                     evaluation_queries=self.evaluation_queries)
        self.node_evaluator.evaluate_all_queried_nodes(evaluation_queries=self.evaluation_queries)

        return tree_expansions

    def update_backward(self, tree_expansions: TreeExpansions):

        update_instructions_batch: upda.UpdateInstructionsBatch
        update_instructions_batch = self.algorithm_node_updater.generate_update_instructions(
            tree_expansions=tree_expansions)

        all_extra_opening_instructions_batch = set()
        while update_instructions_batch:
            node_to_update, update_instructions = update_instructions_batch.popitem()
            extra_update_instructions_batch = self.update_node(node_to_update, update_instructions)
            update_instructions_batch.merge(extra_update_instructions_batch)
        return all_extra_opening_instructions_batch

    def update_node(self, node_to_update, update_instructions):

        ##UPDATES
        new_update_instructions: upda.UpdateInstructions = self.algorithm_node_updater.perform_updates(
            node_to_update=node_to_update,
            update_instructions=update_instructions)

        update_instructions_batch = upda.UpdateInstructionsBatch()
        for parent_node in node_to_update.parent_nodes:
            if parent_node is not None and not new_update_instructions.empty():  # todo is it ever empty?
                assert (parent_node not in update_instructions_batch)
                update_instructions_batch[parent_node] = new_update_instructions

        return update_instructions_batch

    def print_some_stats(self,
                         tree):
        print('Tree stats: move_count', tree.move_count, ' node_count',
              tree.descendants.get_count())
        sum_ = 0
        tree.descendants.print_stats()
        for half_move in tree.descendants:
            sum_ += len(tree.descendants[half_move])
            print('half_move', half_move, len(tree.descendants[half_move]), sum_)

    def print_parents(self, node):
        node_to_print = node
        while node_to_print:
            parents = list(node_to_print.parent_nodes.keys())
            node_to_print = parents[0]

    def test_the_tree(self,
                      tree):
        self.test_count()
        for half_move in tree.descendants:
            for fen in tree.descendants[half_move]:
                node = tree.descendants[half_move][fen]
                node.test()

                # todo add a test for testing if the over match what the board evaluator says!

    def test_count(self,
                   tree):
        assert (tree.root_node.descendants.get_count() == tree.nodes_count)

    def print_best_line(self,
                        tree):
        tree.root_node.minmax_evaluation.print_best_line()