from chessenvironment.boards.board import MyBoard


class ChessEnvironment:

    def __init__(self):
        pass

    def step_create(self, board, move, depth):
        # todo : try to have a very lightweight copy

        if depth < 2: #slow but to be alble to check 3 times repetition at small depth of the tree
            next_board = MyBoard(None, board.chess_board.copy())
        else: # faster
            next_board = MyBoard(None, board.chess_board.copy(stack=False))

        next_board.chess_board.push(move)
        return next_board

    def stepModify(self, board, move):

        board.chess_board.push(move)
