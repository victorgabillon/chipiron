from graphviz import Digraph
import pickle
from .factory import MoveAndValueTree


def add_dot(dot: object, treenode: object) -> object:
    nd = treenode.dot_description()
    dot.node(str(treenode.id), nd)
    for ind, move in enumerate(treenode.moves_children):
        if treenode.moves_children[move] is not None:
            child = treenode.moves_children[move]
            cdd = str(child.id)
            dot.edge(str(treenode.id), cdd, str(move.uci()))
            add_dot(dot, child)


def display_special(node, format, index):
    dot = Digraph(format=format)
    print(';;;', type(node))
    nd = node.dot_description()
    dot.node(str(node.id), nd)
    sorted_moves = [(str(move), move) for move in node.moves_children.keys()]
    sorted_moves.sort()
    for move_key in sorted_moves:
        move = move_key[1]
        child = node.moves_children[move]
        if node.moves_children[move] is not None:
            child = node.moves_children[move]
            cdd = str(child.id)
            edge_description = index[move] + '|' + str(
                move.uci()) + '|' + node.description_tree_visualizer_move(
                child)
            dot.edge(str(node.id), cdd, edge_description)
            dot.node(str(child.id), child.dot_description())
            print('--move:', edge_description)
            print('--child:', child.dot_description())
    return dot


def display(tree: MoveAndValueTree, format):
    dot = Digraph(format=format)
    add_dot(dot, tree.root_node)
    return dot


def save_pdf_to_file(tree: MoveAndValueTree):
    dot = display('pdf')
    round_ = len(tree.root_node.board.move_stack) + 2
    color = 'white' if tree.root_node.player_to_move else 'black'
    dot.render('chipiron/runs/treedisplays/TreeVisual_' + str(int(round_ / 2)) + color + '.pdf')


def save_raw_data_to_file(
        tree: MoveAndValueTree,
        args: dict,
        count='#'):
    round_ = len(tree.root_node.board.move_stack) + 2
    color = 'white' if tree.root_node.player_to_move else 'black'
    filename = 'chipiron/runs/treedisplays/TreeData_' + str(int(round_ / 2)) + color + '-' + str(count) + '.td'
    filename = 'chipiron/debugTreeData_' + str(int(round_ / 2)) + color + '-' + str(count) + '.td'

    import sys
    sys.setrecursionlimit(100000)
    with open(filename, "wb") as f:
        pickle.dump([tree.descendants, tree.root_node], f)
