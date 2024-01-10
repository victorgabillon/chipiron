from typing import Protocol
from chipiron.utils.communication.gui_messages import EvaluationMessage
import queue
from chipiron.environments.chess.board import BoardChi

# VALUE_WHITE_WHEN_OVER is the value_white default value when the node is over
# set atm to be symmetric and high to be preferred
VALUE_WHITE_WHEN_OVER = [VALUE_WHITE_WHEN_OVER_WHITE_WINS, VALUE_WHITE_WHEN_OVER_DRAW,
                         VALUE_WHITE_WHEN_OVER_BLACK_WINS, ] = [1000, 0, -1000]


class BoardEvaluator(Protocol):
    """
    This class evaluates a board
    """

    def value_white(self, board: BoardChi):
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

    def evaluate(self, board: BoardChi):
        print('game evallll')
        evaluation_chi = self.board_evaluator_chi.value_white(board=board)
        evaluation_stock = self.board_evaluator_stock.value_white(board=board)
        print('d',evaluation_stock,evaluation_chi)
        return evaluation_stock, evaluation_chi


class ObservableBoardEvaluator:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info

    game_board_evaluator: GameBoardEvaluator
    mailboxes: list[queue.Queue]

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

    def notify_new_results(self,
                           evaluation_chi,
                           evaluation_stock):
        for mailbox in self.mailboxes:
            message: EvaluationMessage = EvaluationMessage(evaluation_stock=evaluation_stock,
                                                           evaluation_chipiron=evaluation_chi)
            mailbox.put(item=message)

    # forwarding
