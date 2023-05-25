
def tree_strap_one_board(board, tree_value_builder):
    tree_value_builder.explore(board)
    dic_values = get_dic_all_values_of_tree(tree_value_builder.tree)
    return dic_values


def get_dic_all_values_of_tree(tree):
    tree_descendants = tree.root_node.get_descendants()
    dic_values = {}
    for node_descendant in tree_descendants:
        dic_values[node_descendant.board] = node_descendant.get_value_white()
    return dic_values
