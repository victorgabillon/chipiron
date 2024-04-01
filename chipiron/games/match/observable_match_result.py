import copy
import queue
from dataclasses import dataclass, field

from chipiron.games.game.final_game_result import FinalGameResult
from chipiron.utils.communication.gui_messages.gui_messages import MatchResultsMessage
from chipiron.utils.is_dataclass import IsDataclass
from .match_results import MatchResults, SimpleResults


@dataclass
class ObservableMatchResults:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info

    match_results: MatchResults
    mailboxes: 'list[queue.Queue[IsDataclass]]' = field(default_factory=list)

    def subscribe(
            self,
            mailbox: 'queue.Queue[IsDataclass]'
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
            white_player_name_id: str,
            game_result: FinalGameResult
    ) -> None:
        self.match_results.add_result_one_game(
            white_player_name_id,
            game_result
        )
        self.notify_new_results()

    def notify_new_results(
            self
    ) -> None:
        for mailbox in self.mailboxes:
            match_result_copy: MatchResults = self.copy_match_result()
            message: 'MatchResultsMessage' = MatchResultsMessage(
                match_results=match_result_copy
            )
            mailbox.put(item=message)

    # forwarding
    def get_simple_result(
            self
    ) -> SimpleResults:
        simple_results = SimpleResults(
            player_one_wins=self.match_results.get_player_one_wins(),
            player_two_wins=self.match_results.get_player_two_wins(),
            draws=self.match_results.get_draws()
        )
        return simple_results

    def finish(
            self
    ) -> None:
        self.match_results.match_finished = True
        self.notify_new_results()

    def __str__(
            self
    ) -> str:
        return str(self.match_results)
