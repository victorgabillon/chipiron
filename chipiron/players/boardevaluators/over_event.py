"""
Module for handling game over events.
"""

from dataclasses import dataclass
from enum import Enum

import chess


class HowOver(Enum):
    """Represents the possible outcomes of a game.

    Attributes:
        WIN (int): Indicates a win.
        DRAW (int): Indicates a draw.
        DO_NOT_KNOW_OVER (int): Indicates that the outcome is unknown.
    """

    WIN = 1
    DRAW = 2
    DO_NOT_KNOW_OVER = 3


class Winner(Enum):
    """Represents the winner of a chess game.

    Args:
        Enum: The base class for enumeration types.

    Attributes:
        WHITE (Winner): Represents the winner as white.
        BLACK (Winner): Represents the winner as black.
        NO_KNOWN_WINNER (Winner): Represents the absence of a known winner.

    Methods:
        is_none() -> bool: Checks if the winner is None.
        is_white() -> bool: Checks if the winner is white.
        is_black() -> bool: Checks if the winner is black.
    """

    WHITE = chess.WHITE
    BLACK = chess.BLACK
    NO_KNOWN_WINNER = None

    def is_none(self) -> bool:
        """Check if the winner is None.

        Returns:
            bool: True if the winner is None, False otherwise.
        """
        return self is None

    def is_white(self) -> bool:
        """Check if the winner is white.

        Returns:
            bool: True if the winner is white, False otherwise.
        """
        is_white_bool: bool
        if not self.is_none():
            is_white_bool = bool(self) is chess.WHITE
        else:
            is_white_bool = False
        return is_white_bool

    def is_black(self) -> bool:
        """Check if the winner is black.

        Returns:
            bool: True if the winner is black, False otherwise.
        """
        is_black_bool: bool

        if not self.is_none():
            is_black_bool = bool(self) is chess.BLACK
        else:
            is_black_bool = False
        return is_black_bool


class OverTags(str, Enum):
    """Represents the possible tags for game over events.

    Attributes:
        TAG_WIN_WHITE (str): Tag indicating a win for the white player.
        TAG_WIN_BLACK (str): Tag indicating a win for the black player.
        TAG_DRAW (str): Tag indicating a draw.
        TAG_DO_NOT_KNOW (str): Tag indicating an unknown outcome.
    """

    TAG_WIN_WHITE = "Win-Wh"
    TAG_WIN_BLACK = "Win-Bl"
    TAG_DRAW = "Draw"
    TAG_DO_NOT_KNOW = "?"


@dataclass(slots=True)
class OverEvent:
    """Represents an event that indicates the end of a game.

    Attributes:
        how_over (HowOver): The way the game ended.
        who_is_winner (Winner): The winner of the game.

    Raises:
        AssertionError: If the `how_over` attribute is not a valid value from the `HowOver` enum.
        AssertionError: If the `who_is_winner` attribute is not a valid value from the `Winner` enum.
        Exception: If the winner is not properly defined.

    Methods:
        __post_init__: Performs additional initialization after the object is created.
        becomes_over: Sets the `how_over` and `who_is_winner` attributes.
        get_over_tag: Returns a tag string used in databases.
        __bool__: Raises an exception.
        is_over: Checks if the game is over.
        is_win: Checks if the game ended with a win.
        is_draw: Checks if the game ended with a draw.
        is_winner: Checks if the specified player is the winner.
        print_info: Prints information about the `OverEvent` object.
        test: Performs tests on the `OverEvent` object.
    """

    how_over: HowOver = HowOver.DO_NOT_KNOW_OVER
    who_is_winner: Winner = Winner.NO_KNOWN_WINNER
    termination: chess.Termination | None = None

    def __post_init__(self) -> None:
        assert self.how_over in HowOver
        assert self.who_is_winner in Winner

        if self.how_over == HowOver.WIN:
            assert (
                self.who_is_winner is Winner.WHITE or self.who_is_winner is Winner.BLACK
            )
        elif self.how_over == HowOver.DRAW:
            assert self.who_is_winner is Winner.NO_KNOWN_WINNER

    def becomes_over(
        self,
        how_over: HowOver,
        termination: chess.Termination | None,
        who_is_winner: Winner = Winner.NO_KNOWN_WINNER,
    ) -> None:
        """Sets the `how_over` and `who_is_winner` attributes.

        Args:
            how_over (HowOver): The way the game ended.
            who_is_winner (Winner, optional): The winner of the game. Defaults to `Winner.NO_KNOWN_WINNER`.
        """
        self.how_over = how_over
        self.who_is_winner = who_is_winner
        self.termination = termination

    def get_over_tag(self) -> OverTags:
        """Returns a tag string used in databases.

        Returns:
            OverTags: The tag string representing the game outcome.

        Raises:
            Exception: If the winner is not properly defined.
        """
        over_tag: OverTags
        if self.how_over == HowOver.WIN:
            if self.who_is_winner.is_white():
                over_tag = OverTags.TAG_WIN_WHITE
            elif self.who_is_winner.is_black():
                over_tag = OverTags.TAG_WIN_BLACK
            else:
                raise ValueError("error: winner is not properly defined.")
        elif self.how_over == HowOver.DRAW:
            over_tag = OverTags.TAG_DRAW
        elif self.how_over == HowOver.DO_NOT_KNOW_OVER:
            over_tag = OverTags.TAG_DO_NOT_KNOW
        else:
            raise ValueError("error: over is not properly defined.")
        return over_tag

    def __bool__(self) -> None:
        """Raises an exception.

        Raises:
            Exception: Always raises an exception.
        """
        raise ValueError("Nooooooooooo  in over ebvent.py")

    def is_over(self) -> bool:
        """Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.how_over in {HowOver.WIN, HowOver.DRAW}

    def is_win(self) -> bool:
        """Checks if the game ended with a win.

        Returns:
            bool: True if the game ended with a win, False otherwise.
        """
        return self.how_over == HowOver.WIN

    def is_draw(self) -> bool:
        """Checks if the game ended with a draw.

        Returns:
            bool: True if the game ended with a draw, False otherwise.
        """
        return self.how_over == HowOver.DRAW

    def is_winner(self, player: chess.Color) -> bool:
        """Checks if the specified player is the winner.

        Args:
            player (chess.Color): The player to check.

        Returns:
            bool: True if the specified player is the winner, False otherwise.

        Raises:
            AssertionError: If the `player` argument is not a valid value from the `chess.Color` enum.
        """
        assert player in {chess.WHITE, chess.BLACK}

        is_winner: bool
        if self.how_over == HowOver.WIN:
            is_winner = bool(
                self.who_is_winner == Winner.WHITE
                and player == chess.WHITE
                or self.who_is_winner == Winner.BLACK
                and player == chess.BLACK
            )
        else:
            is_winner = False
        return is_winner

    def print_info(self) -> None:
        """Prints information about the `OverEvent` object."""
        print(
            "over_event:",
            "how_over:",
            self.how_over,
            "who_is_winner:",
            self.who_is_winner,
        )

    def test(self) -> None:
        """Performs tests on the `OverEvent` object."""
        if self.how_over == HowOver.WIN:
            assert self.who_is_winner is not None
            assert self.who_is_winner.is_white() or self.who_is_winner.is_black()
        if self.how_over == HowOver.DRAW:
            assert self.who_is_winner is None
