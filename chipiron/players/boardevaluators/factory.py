from chipiron.players.boardevaluators.stockfish_board_evaluator import StockfishBoardEvaluator
from chipiron.players.boardevaluators.board_evaluator import ObservableBoardEvaluator
from chipiron.players.boardevaluators.board_evaluator import BoardEvaluator


class BoardEvaluatorFactory:

    def create(self) -> BoardEvaluator:
        board_evaluator: BoardEvaluator = StockfishBoardEvaluator()
        return board_evaluator


class ObservableBoardEvaluatorFactory(BoardEvaluatorFactory):
    def __init__(self):
        self.subscribers = []

    def create(self):
        board_evaluator: BoardEvaluator = super().create()
        if self.subscribers:

            board_evaluator = ObservableBoardEvaluator(board_evaluator)
            for subscriber in self.subscribers:
                board_evaluator.subscribe(subscriber)
        return board_evaluator

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
