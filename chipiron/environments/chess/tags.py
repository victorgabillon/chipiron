from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChessStartTag:
    fen: str
