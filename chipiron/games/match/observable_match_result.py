import copy
import queue
import typing
from chipiron.utils.communication.gui_messages import MatchResultsMessage
from dataclasses import dataclass
import chess
from .math_results import MatchResults


class ObservableMatchResults:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info

    match_results: MatchResults
    mailboxes: typing.List[queue.Queue]

    def __init__(self, match_result: MatchResults) -> None:
        self.match_result = match_result
        self.mailboxes = []

    def subscribe(self, mailbox: queue.Queue) -> None:
        self.mailboxes.append(mailbox)

    def copy_match_result(self) -> MatchResults:
        match_result_copy: MatchResults = copy.deepcopy(self.match_result)
        return match_result_copy

    # wrapped function
    def add_result_one_game(self, white_player_name_id, game_result) -> None:
        self.match_result.add_result_one_game(white_player_name_id, game_result)
        self.notify_new_results()

    def notify_new_results(self) -> None:
        for mailbox in self.mailboxes:
            match_result_copy: MatchResults = self.copy_match_result()
            message: MatchResultsMessage = MatchResultsMessage(
                match_results=match_result_copy
            )
            mailbox.put(item=message)

    # forwarding
    def get_simple_result(self):
        return self.match_result.get_player_one_wins(), \
            self.match_result.get_player_two_wins(), \
            self.match_result.get_draws()

    def finish(self):
        self.match_result.match_finished = True
        self.notify_new_results()


@dataclass
class MatchReport:
    match_results: MatchResults
    match_move_history: dict[int, list[chess.Move]]
