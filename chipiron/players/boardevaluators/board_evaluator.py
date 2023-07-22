from typing import Protocol

# VALUE_WHITE_WHEN_OVER is the value_white default value when the node is over
# set atm to be symmetric and high to be preferred
VALUE_WHITE_WHEN_OVER = [VALUE_WHITE_WHEN_OVER_WHITE_WINS, VALUE_WHITE_WHEN_OVER_DRAW,
                         VALUE_WHITE_WHEN_OVER_BLACK_WINS, ] = [1000, 0, -1000]


class BoardEvaluator(Protocol):
    """
    This class evaluates a board
    """

    def value_white(self, board):
        """Evaluates a board"""
        ...


class GameBoardEvaluator:
    """
    This class is a collection of evaluator that display their analysis during the game.
    They are not players just external analysis and display
    """
    board_evaluator_stock: BoardEvaluator
    board_evaluator_chi: BoardEvaluator

    def __init__(self,
                 board_evaluator_stock: BoardEvaluator,
                 board_evaluator_chi: BoardEvaluator
                 ):
        self.board_evaluator_stock = board_evaluator_stock
        self.board_evaluator_chi = board_evaluator_chi

    def evaluate(self, board):
        evaluation_chi = self.board_evaluator_chi.value_white(board=board)
        evaluation_stock = self.board_evaluator_stock.value_white(board=board)

        return evaluation_stock, evaluation_chi


class ObservableBoardEvaluator:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info

    game_board_evaluator: GameBoardEvaluator

    def __init__(self,
                 game_board_evaluator: GameBoardEvaluator
                 ):
        self.game_board_evaluator = game_board_evaluator

        self.mailboxes = []

    def subscribe(self, mailbox):
        self.mailboxes.append(mailbox)

    # wrapped function
    def evaluate(self, board):
        evaluation_stock, evaluation_chi = self.game_board_evaluator.evaluate(board=board)

        self.notify_new_results(evaluation_chi, evaluation_stock)
        return evaluation_stock, evaluation_chi,

    def notify_new_results(self, evaluation_chi, evaluation_stock):
        for mailbox in self.mailboxes:
            mailbox.put({'type': 'evaluation', 'evaluation_stock': evaluation_stock, 'evaluation_chi': evaluation_chi})

    # forwarding


class BoardEvaluatorWrapped:
    ''' wrapping board evaluator with syzygy '''
    node_evaluator: object
    syzygy_evaluator: object

    def __init__(self, node_evaluator, syzygy):
        self.node_evaluator = node_evaluator
        self.syzygy_evaluator = syzygy

    def value_white(self, node):
        value_white = self.syzygy_value_white(node.board)
        if value_white is None:
            value_white = self.node_evaluator.value_white(node)
        return value_white

    def syzygy_value_white(self, board):
        if self.syzygy_evaluator is None or not self.syzygy_evaluator.fast_in_table(board):
            return None
        else:
            return self.syzygy_evaluator.value_white(board)

    def check_obvious_over_events(self, node):
        """ updates the node.over object
         if the game is obviously over"""
        game_over = node.tree_node.board.is_game_over()
        if game_over:
            value_as_string = node.board.result()
            if value_as_string == '0-1':
                how_over_ = node.minmax_evaluation.over_event.WIN
                who_is_winner_ = chess.BLACK
            elif value_as_string == '1-0':
                how_over_ = node.minmax_evaluation.over_event.WIN
                who_is_winner_ = chess.WHITE
            elif value_as_string == '1/2-1/2':
                how_over_ = node.minmax_evaluation.over_event.DRAW
                who_is_winner_ = node.minmax_evaluation.over_event.NO_KNOWN_WINNER
            node.minmax_evaluation.over_event.becomes_over(how_over=how_over_,
                                                           who_is_winner=who_is_winner_)

        elif self.syzygy_evaluator and self.syzygy_evaluator.fast_in_table(node.tree_node.board):
            self.syzygy_evaluator.set_over_event(node)

    def value_white_from_over_event(self, over_event):
        """ returns the value white given an over event"""
        assert over_event.is_over()
        if over_event.is_win():
            assert (not over_event.is_draw())
            if over_event.is_winner(chess.WHITE):
                return VALUE_WHITE_WHEN_OVER_WHITE_WINS
            else:
                return VALUE_WHITE_WHEN_OVER_BLACK_WINS
        else:  # draw
            assert (over_event.is_draw())
            return VALUE_WHITE_WHEN_OVER_DRAW
