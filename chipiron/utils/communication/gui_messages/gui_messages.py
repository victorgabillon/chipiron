"""
Module that contains the GUI messages that are sent to the GUI from the game.
"""

from dataclasses import dataclass

from chipiron.games.match.match_results import MatchResults


@dataclass
class MatchResultsMessage:
    """
    Represents a message containing the match results that are sent to the GUI from the game.

    Attributes:
        match_results (MatchResults): The match results object containing the details of the match.
    """

    match_results: MatchResults
