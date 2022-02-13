from typing import Protocol


class BoardEvaluator(Protocol):
    """
    This class evaluates a board
    """

    def evaluate(self, board):
        """Evaluates a board"""
        ...


class ObservableBoardEvaluator:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info
    def __init__(self, board_evaluator):
        self.board_evaluator = board_evaluator
        self.mailboxes = []

    def subscribe(self, mailbox):
        self.mailboxes.append(mailbox)

    # wrapped function
    def evaluate(self, board):
        evaluation = self.board_evaluator.evaluate(board)
        self.notify_new_results(evaluation)
        return evaluation

    def notify_new_results(self, evaluation):
        for mailbox in self.mailboxes:
            mailbox.put({'type': 'evaluation', 'evaluation': evaluation})

    # forwarding
