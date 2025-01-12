"""
Module to create a MatchResults object.

This module provides a MatchResultsFactory class that is responsible for creating MatchResults objects.
It also provides a way to subscribe to the MatchResults object and receive updates.

Classes:
- MatchResultsFactory: A factory class to create MatchResults objects and manage subscribers.

"""

import queue

from chipiron.games.match.match_results import IMatchResults, MatchResults
from chipiron.games.match.observable_match_result import ObservableMatchResults
from chipiron.utils.dataclass import IsDataclass


class MatchResultsFactory:
    """A factory class for creating MatchResults objects.

    This class provides methods to create MatchResults objects and subscribe subscribers to receive match results.

    Attributes:
        player_one_name (str): The name of player one.
        player_two_name (str): The name of player two.
        subscribers (list[queue.Queue[IsDataclass]]): A list of subscribers to receive match results.

    Methods:
        __init__(self, player_one_name: str, player_two_name: str) -> None:
            Initialize the MatchResultsFactory.

        create(self) -> IMatchResults:
            Create a MatchResults object.

        subscribe(self, subscriber: queue.Queue[IsDataclass]) -> None:
            Subscribe a subscriber to receive match results.
    """

    player_one_name: str
    player_two_name: str
    subscribers: list[queue.Queue[IsDataclass]] = []

    def __init__(self, player_one_name: str, player_two_name: str) -> None:
        """
        Initialize the MatchResultsFactory.

        Args:
            player_one_name (str): The name of player one.
            player_two_name (str): The name of player two.
        """
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.subscribers = []

    def create(self) -> IMatchResults:
        """
        Create a MatchResults object.

        Returns:
            IMatchResults: The created MatchResults object.
        """
        match_result: MatchResults = MatchResults(
            player_one_name_id=self.player_one_name,
            player_two_name_id=self.player_two_name,
        )
        if self.subscribers:
            obs_match_result: ObservableMatchResults = ObservableMatchResults(
                match_result
            )
            for subscriber in self.subscribers:
                obs_match_result.subscribe(subscriber)
            return obs_match_result
        else:
            return match_result

    def subscribe(self, subscriber: queue.Queue[IsDataclass]) -> None:
        """Subscribe a subscriber to receive match results.

        Args:
            subscriber (queue.Queue[IsDataclass]): The subscriber to be added to the list of subscribers.
        """
        self.subscribers.append(subscriber)
