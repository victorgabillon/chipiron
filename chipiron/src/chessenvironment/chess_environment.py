from src.chessenvironment.boards.board import MyBoard


class ChessEnvironment:

    def __init__(self):
        pass

    def step_create(self, board, move, depth):
        # todo : try to have a very lightweight copy

        if depth < 2:  # slow but to be able to check 3 times repetition at small depth of the tree
            next_board = MyBoard(chess_board=board.chess_board.copy(stack=True))
        else:  # faster
            next_board = MyBoard(chess_board=board.chess_board.copy(stack=False))

        board_modifications = next_board.push_and_return_modification(move)

        return next_board, board_modifications

    def step_modify(self, board, move):
        board.chess_board.push(move)
