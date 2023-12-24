from players.move_selector.treevalue.trees.move_and_value_tree import MoveAndValueTree
from players.move_selector.treevalue.nodes.tree_node_with_proportions import ProportionsNode


class VertiZipfTree(MoveAndValueTree):
    def create_tree_node(self, board, half_move, count, father_node):
        return ProportionsNode(board, half_move, count, father_node, self.zipf_style)


class VertiZipf:

    def __init__(self, arg):
        super().__init__(arg)

    def zipfDistribution(self, K):
        res = [0] * K
        sum_ = 0
        for i in range(1, K + 1):
            sum_ = sum_ + float(1 / i)
        for i in range(K):
            res[i] = float(1 / (i + 1)) / sum_
        # print('vvvv', res, sum(res))
        return res

    def create_tree(self, board):
        return VertiZipfTree(self.environment, self.board_evaluator, self.color_player, self.arg, board)

    def choose_node_and_move_to_open(self):
        # print('----------------')
        node = self.tree.root_node
        found = False
        while not found:
            # print('+++++++++++++++++++++')

            bestSeq = node.best_node_sequence_filtered_from_over()
            K = len(bestSeq)
            zipf = self.zipfDistribution(K)
            #  print('p',zipf,K)
            bestK = 0
            bestValue = self.visitsDepth[K][0] / zipf[0]
            for i in range(K):
                value = self.visitsDepth[K][i] / zipf[i]
                if value < bestValue:
                    bestValue = value
                    bestK = i

            self.visitsDepth[K][bestK] += 1
            # print('tc',node == self.tree.rootNode)

            node = bestSeq[bestK]
            # print('t', K, bestK, node.depth, bestSeq, node == self.tree.rootNode)
            #   for i in range(10):
            #      print(self.visitsDepth[i])
            # input("Press Enter to continue...")

            #  print('o',node == self.tree.rootNode)
            if node.moves_children == {}:
                #  print('j')
                #    print('fg',node,bestSeq,node.children)
                #    print('jd', node.bestMoveSequence, node.over)
                assert (not node.is_over())
                assert (node.moves_children == {})
                break

            node = node.choose_child_with_visits_and_proportions()
        return self.opening_instructor.instructions_to_open_all_moves(node)

    def get_move_from_player(self, board, timetoMove):

        print(board.chess_board)

        self.visitsDepth = []  # todo put in one line no?
        for K in range(1000):
            self.visitsDepth.append([])
            for i in range(K):
                self.visitsDepth[K].append(0)

        return super().get_move_from_player(board, timetoMove)

    def print_info(self):
        super().print_info()
        print('VertiZipf')
