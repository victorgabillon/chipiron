from chipiron.games.game.game_manager import GameManager, FinalGameResult
import copy
import queue
import typing


class MatchResults:

    def __init__(self, player_one_id, player_two_id):
        self.player_one_name_id = player_one_id
        self.player_two_name_id = player_two_id
        self.number_of_games = 0
        self.player_one_is_white_white_wins = 0
        self.player_one_is_white_black_wins = 0
        self.player_one_is_white_draws = 0
        self.player_two_is_white_white_wins = 0
        self.player_two_is_white_black_wins = 0
        self.player_two_is_white_draws = 0

    def get_player_one_wins(self):
        return self.player_one_is_white_white_wins + self.player_two_is_white_black_wins

    def get_player_two_wins(self):
        return self.player_one_is_white_black_wins + self.player_two_is_white_white_wins

    def get_draws(self):
        return self.player_one_is_white_draws + self.player_two_is_white_draws

    def get_simple_result(self):
        return self.get_player_one_wins(), self.get_player_two_wins(), self.get_draws()

    def add_result_one_game(self, white_player_name_id, game_result):
        self.number_of_games += 1
        if white_player_name_id == self.player_one_name_id:
            if game_result == FinalGameResult.WIN_FOR_WHITE:
                self.player_one_is_white_white_wins += 1
            elif game_result == FinalGameResult.WIN_FOR_BLACK:
                self.player_one_is_white_black_wins += 1
            elif game_result == FinalGameResult.DRAW:
                self.player_one_is_white_draws += 1
            else:
                pass
                # raise Exception('#')
        elif white_player_name_id == self.player_two_name_id:
            if game_result == FinalGameResult.WIN_FOR_WHITE:
                self.player_two_is_white_white_wins += 1
            elif game_result == FinalGameResult.WIN_FOR_BLACK:
                self.player_two_is_white_black_wins += 1
            elif game_result == FinalGameResult.DRAW:
                self.player_two_is_white_draws += 1
            else:
                raise Exception('!')
        else:
            raise Exception('?')

    def __str__(self):
        str_ = 'Main result: ' + self.player_one_name_id + ' wins ' + str(self.get_player_one_wins()) + ' '
        str_ += self.player_two_name_id + ' wins ' + str(self.get_player_two_wins())
        str_ += ' draws ' + str(self.get_draws()) + '\n'

        str_ += self.player_one_name_id + ' with white: '
        str_ += 'Wins ' + str(self.player_one_is_white_white_wins)
        str_ += ', Losses ' + str(self.player_one_is_white_black_wins)
        str_ += ', Draws ' + str(self.player_one_is_white_draws)
        str_ += '\n           with black: '
        str_ += 'Wins ' + str(self.player_two_is_white_black_wins)
        str_ += ', Losses ' + str(self.player_two_is_white_white_wins)
        str_ += ', Draws ' + str(self.player_two_is_white_draws) + '\n'

        str_ += self.player_two_name_id + ' with white: '
        str_ += 'Wins ' + str(self.player_two_is_white_white_wins)
        str_ += ', Losses ' + str(self.player_two_is_white_black_wins)
        str_ += ', Draws ' + str(self.player_two_is_white_draws)
        str_ += '\n           with black: '
        str_ += 'Wins ' + str(self.player_one_is_white_black_wins)
        str_ += ', Losses ' + str(self.player_one_is_white_white_wins)
        str_ += ', Draws ' + str(self.player_one_is_white_draws)
        return str_


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
            mailbox.put({'type': 'match_results', 'match_results': match_result_copy})

    # forwarding
    def get_simple_result(self):
        return self.match_result.get_player_one_wins(), \
               self.match_result.get_player_two_wins(), \
               self.match_result.get_draws()
