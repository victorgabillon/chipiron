# from players.treevalue.trees. import NoisyValueTree
#
#
# class OpeningBook:
#     self.required_number_of_games =1000
#     def __init__(self):
#         self.tree = NoisyValueTree()
#         self.node_picker
#
#     def get_opening_position_to_learn(self):
#         node = self.tree.root_node
#
#         while True:
#
#
#     def position_result(self,starting_position, p1wins, p2wins, draws):
#         pass


#
# class OpeningBook:
#     def __init__(self, required_number_of_games=1000):
#         """
#         Initialize the OpeningBook with a NoisyValueTree and required games per node.
#         """
#         self.required_number_of_games = required_number_of_games
#         self.tree = NoisyValueTree()  # Tree structure to store opening moves
#         self.node_picker = self.tree.get_node_picker()  # Tree search algorithm
#
#     def get_opening_position_to_learn(self):
#         """
#         Select a node (position) from the tree to evaluate.
#         """
#         node = self.tree.root_node  # Start from the root node
#
#         while True:
#             # Use the node picker to select the next node to evaluate
#             node = self.node_picker.select(node)
#
#             # Check if the node needs more evaluations
#             if node.evaluations < self.required_number_of_games:
#                 return node
#
#     def position_result(self, starting_position, p1wins, p2wins, draws):
#         """
#         Update the tree with the results of evaluating a position.
#         :param starting_position: The position being evaluated.
#         :param p1wins: Number of wins for player 1.
#         :param p2wins: Number of wins for player 2.
#         :param draws: Number of draws.
#         """
#         # Find the node corresponding to the starting position
#         node = self.tree.find_node(starting_position)
#
#         if node is None:
#             raise ValueError("Position not found in the tree.")
#
#         # Update the node's statistics
#         node.update_statistics(p1wins, p2wins, draws)
