
def are_all_moves_and_children_opened(tree_node)-> bool:
    return tree_node.all_legal_moves_generated and tree_node.non_opened_legal_moves == set()
