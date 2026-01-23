from enum import Enum


# --------- Enums / IDs ---------
class GameKind(str, Enum):
    CHESS = "chess"
    CHECKERS = "checkers"
