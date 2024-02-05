import copy
import queue
import typing
from chipiron.utils.communication.gui_messages import MatchResultsMessage
from dataclasses import dataclass, field
import chess
from .math_results import MatchResults


@dataclass
class ObservableMatchResults:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info

    match_results: MatchResults
    mailboxes: list[queue.Queue] = field(default_factory=list)

    def subscribe(
            self,
            mailbox: queue.Queue
    ) -> None:
        self.mailboxes.append(mailbox)

    def copy_match_result(
            self
    ) -> MatchResults:
        match_result_copy: MatchResults = copy.deepcopy(self.match_results)
        return match_result_copy

    # wrapped function
    def add_result_one_game(
            self,
            white_player_name_id,
            game_result
    ) -> None:
        self.match_results.add_result_one_game(white_player_name_id, game_result)
        self.notify_new_results()

    def notify_new_results(
            self
    ) -> None:
        for mailbox in self.mailboxes:
            match_result_copy: MatchResults = self.copy_match_result()
            message: MatchResultsMessage = MatchResultsMessage(
                match_results=match_result_copy
            )
            mailbox.put(item=message)

    # forwarding
    def get_simple_result(
            self
    ):
        return self.match_results.get_player_one_wins(), \
            self.match_results.get_player_two_wins(), \
            self.match_results.get_draws()

    def finish(
            self
    ):
        self.match_results.match_finished = True
        self.notify_new_results()

    def __str__(
            self
    ) -> str:
        return str(self.match_results)


@dataclass
class MatchReport:
    match_results: MatchResults
    match_move_history: dict[int, list[chess.Move]]
