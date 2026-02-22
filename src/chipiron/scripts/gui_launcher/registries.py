"""Game-specific registries used by the GUI."""

from dataclasses import dataclass

from chipiron.environments.types import GameKind
from chipiron.players.player_ids import PlayerConfigTag


@dataclass(frozen=True)
class PlayerOption:
    """Displayable player option for a given game."""

    label: str
    tag: PlayerConfigTag
    supports_strength: bool


CHESS_PLAYER_OPTIONS: list[PlayerOption] = [
    PlayerOption("RecurZipfBase3", PlayerConfigTag.RECUR_ZIPF_BASE_3, True),
    PlayerOption("Uniform", PlayerConfigTag.UNIFORM, True),
    PlayerOption("Sequool", PlayerConfigTag.SEQUOOL, True),
    PlayerOption("Human Player", PlayerConfigTag.GUI_HUMAN, False),
]

CHECKERS_PLAYER_OPTIONS: list[PlayerOption] = [
    PlayerOption("Human Player", PlayerConfigTag.GUI_HUMAN, False),
]


def player_options_for_game(game_kind: GameKind) -> list[PlayerOption]:
    """Return allowed players for a game kind."""
    match game_kind:
        case GameKind.CHESS:
            return CHESS_PLAYER_OPTIONS
        case GameKind.CHECKERS:
            return CHECKERS_PLAYER_OPTIONS
        case _:
            return [PlayerOption("Human Player", PlayerConfigTag.GUI_HUMAN, False)]  # type: ignore[unreachable]


CHESS_STARTING_POSITIONS: dict[str, str] = {
    "Standard": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "End game": "6k1/p7/8/8/7N/7K/2N5/8 w - - 0 1",
    "Crushing winning (White)": "5nk1/3ppppp/8/r7/3RR3/5QRR/6PP/5NK1 w - - 0 1",
}


def starting_positions_for_game(game_kind: GameKind) -> dict[str, str]:
    """Return UI starting-position choices by game kind."""
    match game_kind:
        case GameKind.CHESS:
            return CHESS_STARTING_POSITIONS
        case GameKind.CHECKERS:
            return {"Standard": "STANDARD"}
        case _:
            return {"Standard": "STANDARD"}  # type: ignore[unreachable]
