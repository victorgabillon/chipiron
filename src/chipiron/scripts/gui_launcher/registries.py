"""Game-specific registries used by the GUI launcher."""

from dataclasses import dataclass

from chipiron.environments.types import GameKind
from chipiron.players.player_ids import PlayerConfigTag

from .participant_selection import ParticipantSelection


class UnknownPlayerLabelError(ValueError):
    """Raised when a launcher label cannot be resolved for a game."""

    def __init__(self, *, game_kind: GameKind, label: str) -> None:
        """Initialize the error with the unresolved launcher label."""
        super().__init__(f"Unknown player label {label!r} for {game_kind.value}")


@dataclass(frozen=True, slots=True)
class PlayerOption:
    """Displayable player option for a given game."""

    label: str
    tag: PlayerConfigTag
    supports_strength: bool


@dataclass(frozen=True, slots=True)
class LauncherSpec:
    """Declarative launcher behavior for a game kind."""

    participant_count: int
    participant_labels: tuple[str, ...]
    player_options: tuple[PlayerOption, ...]
    starting_positions: dict[str, str]
    default_starting_position_key: str
    default_participants: tuple[ParticipantSelection, ...]


CHESS_PLAYER_OPTIONS: tuple[PlayerOption, ...] = (
    PlayerOption("RecurZipfBase3", PlayerConfigTag.RECUR_ZIPF_BASE_3, True),
    PlayerOption("Uniform", PlayerConfigTag.UNIFORM, True),
    PlayerOption("Sequool", PlayerConfigTag.SEQUOOL, True),
    PlayerOption("Human Player", PlayerConfigTag.GUI_HUMAN, False),
)

CHECKERS_PLAYER_OPTIONS: tuple[PlayerOption, ...] = (
    PlayerOption("Human Player", PlayerConfigTag.GUI_HUMAN, False),
    PlayerOption("Random", PlayerConfigTag.RANDOM, False),
    PlayerOption("Tree (piece count)", PlayerConfigTag.CHECKERS_TREE_PIECECOUNT, False),
)

INTEGER_REDUCTION_PLAYER_OPTIONS: tuple[PlayerOption, ...] = (
    PlayerOption("Human Player", PlayerConfigTag.GUI_HUMAN, False),
    PlayerOption("Random", PlayerConfigTag.RANDOM, False),
    PlayerOption(
        "Tree (basic eval)",
        PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC,
        False,
    ),
    PlayerOption(
        "Tree (basic eval + debug)",
        PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC_DEBUG,
        False,
    ),
)

MORPION_PLAYER_OPTIONS: tuple[PlayerOption, ...] = (
    PlayerOption("Human Player", PlayerConfigTag.GUI_HUMAN, False),
    PlayerOption("Random", PlayerConfigTag.RANDOM, False),
    PlayerOption("Tree (basic eval)", PlayerConfigTag.MORPION_TREE_BASIC, False),
    PlayerOption(
        "Tree (uniform depth 2 + debug)",
        PlayerConfigTag.MORPION_UNIFORM_DEPTH_2_DEBUG,
        False,
    ),
)

CHESS_STARTING_POSITIONS: dict[str, str] = {
    "Standard": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "End game": "6k1/p7/8/8/7N/7K/2N5/8 w - - 0 1",
    "Crushing winning (White)": "5nk1/3ppppp/8/r7/3RR3/5QRR/6PP/5NK1 w - - 0 1",
}

_LAUNCHER_SPECS: dict[GameKind, LauncherSpec] = {
    GameKind.CHESS: LauncherSpec(
        participant_count=2,
        participant_labels=("White", "Black"),
        player_options=CHESS_PLAYER_OPTIONS,
        starting_positions=CHESS_STARTING_POSITIONS,
        default_starting_position_key="Standard",
        default_participants=(
            ParticipantSelection(
                player_tag=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                strength=1,
            ),
            ParticipantSelection(
                player_tag=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                strength=1,
            ),
        ),
    ),
    GameKind.CHECKERS: LauncherSpec(
        participant_count=2,
        participant_labels=("White", "Black"),
        player_options=CHECKERS_PLAYER_OPTIONS,
        starting_positions={"Standard": "STANDARD"},
        default_starting_position_key="Standard",
        default_participants=(
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
        ),
    ),
    GameKind.INTEGER_REDUCTION: LauncherSpec(
        participant_count=1,
        participant_labels=("Solo",),
        player_options=INTEGER_REDUCTION_PLAYER_OPTIONS,
        starting_positions={"Small": "7", "Standard": "15", "Large": "31"},
        default_starting_position_key="Standard",
        default_participants=(
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
        ),
    ),
    GameKind.MORPION: LauncherSpec(
        participant_count=1,
        participant_labels=("Solo",),
        player_options=MORPION_PLAYER_OPTIONS,
        starting_positions={"Standard": "5T"},
        default_starting_position_key="Standard",
        default_participants=(
            ParticipantSelection(player_tag=PlayerConfigTag.GUI_HUMAN),
        ),
    ),
}


def launcher_spec_for_game(game_kind: GameKind) -> LauncherSpec:
    """Return launcher behavior for the selected game."""
    return _LAUNCHER_SPECS[game_kind]


def player_options_for_game(game_kind: GameKind) -> list[PlayerOption]:
    """Return allowed players for a game kind."""
    return list(launcher_spec_for_game(game_kind).player_options)


def starting_positions_for_game(game_kind: GameKind) -> dict[str, str]:
    """Return UI starting-position choices by game kind."""
    return dict(launcher_spec_for_game(game_kind).starting_positions)


def player_option_for_label(game_kind: GameKind, label: str) -> PlayerOption:
    """Resolve a player option from its display label."""
    for option in launcher_spec_for_game(game_kind).player_options:
        if option.label == label:
            return option
    raise UnknownPlayerLabelError(game_kind=game_kind, label=label)


def player_label_for_tag(game_kind: GameKind, player_tag: PlayerConfigTag) -> str:
    """Resolve the display label for a player tag, with a safe fallback."""
    launcher_spec = launcher_spec_for_game(game_kind)
    for option in launcher_spec.player_options:
        if option.tag is player_tag:
            return option.label

    for option in launcher_spec.player_options:
        if option.tag is PlayerConfigTag.GUI_HUMAN:
            return option.label

    return launcher_spec.player_options[0].label
