from chipiron.games.game.final_game_result import FinalGameResult
from dataclasses import dataclass
import chess


@dataclass
class MatchResults:
    player_one_name_id: str
    player_two_name_id: str
    number_of_games: int = 0
    player_one_is_white_white_wins: int = 0
    player_one_is_white_black_wins: int = 0
    player_one_is_white_draws: int = 0
    player_two_is_white_white_wins: int = 0
    player_two_is_white_black_wins: int = 0
    player_two_is_white_draws: int = 0
    match_finished: bool = False

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

    def finish(self):
        self.match_finished = True

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


@dataclass
class MatchReport:
    match_results: MatchResults
    match_move_history: dict[int, list[chess.Move]]
