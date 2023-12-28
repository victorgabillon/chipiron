from enum import Enum


class MoveSelectorTypes(str, Enum):
    RandomPlayer: str = 'RandomPlayer'
    TreeAndValue: str = 'TreeAndValue'
    Stockfish: str = 'Stockfish'
    Human: str = 'Human'
