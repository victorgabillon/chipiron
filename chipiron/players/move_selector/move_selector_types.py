from enum import Enum

from . import treevalue
from . import human


class MoveSelectorTypes(str, Enum):
    Random: str = 'Random'
    TreeAndValue: str = treevalue.Tree_Value_Name_Literal
    Stockfish: str = 'Stockfish'
    Human: str = human.Human_Name_Literal
