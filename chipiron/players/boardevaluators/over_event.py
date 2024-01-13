import chess
from dataclasses import dataclass
from enum import Enum


class HowOver(Enum):
    WIN = 1
    DRAW = 2
    DO_NOT_KNOW_OVER = 3


class Winner(Enum):
    WHITE = chess.WHITE
    BLACK = chess.BLACK
    NO_KNOWN_WINNER = None


class OverTags(str, Enum):
    TAG_WIN_WHITE = 'Win-Wh'
    TAG_WIN_BLACK = 'Win-Bl'
    TAG_DRAW = 'Draw'
    TAG_DO_NOT_KNOW = '?'


@dataclass(slots=True)
class OverEvent:
    how_over: HowOver = HowOver.DO_NOT_KNOW_OVER
    who_is_winner: Winner = Winner.NO_KNOWN_WINNER

    def __post_init__(self):
        assert self.how_over in HowOver
        assert self.who_is_winner in Winner

        if self.how_over == HowOver.WIN:
            assert (self.who_is_winner is Winner.WHITE or self.who_is_winner is Winner.BLACK)
        elif self.how_over == HowOver.DRAW:
            assert (self.who_is_winner is Winner.NO_KNOWN_WINNER)

    def becomes_over(self, how_over, who_is_winner=Winner.NO_KNOWN_WINNER):
        self.how_over = how_over
        self.who_is_winner = who_is_winner


    def get_over_tag(self):
        """ returns a simple string that is used a tag in the databases"""
        if self.how_over == HowOver.WIN:
            if self.who_is_winner == chess.WHITE:
                return OverTags.TAG_WIN_WHITE
            elif self.who_is_winner == chess.BLACK:
                return OverTags.TAG_WIN_BLACK
            else:
                raise Exception('error: winner is not properly defined.')
        elif self.how_over == HowOver.DRAW:
            return OverTags.TAG_DRAW
        elif self.how_over == HowOver.DO_NOT_KNOW_OVER:
            return OverTags.TAG_DO_NOT_KNOW
        else:
            raise Exception('error: over is not properly defined.')

    def __bool__(self):
        assert (1 == 0)

    def is_over(self) -> bool:
        return self.how_over == HowOver.WIN or self.how_over == HowOver.DRAW

    def is_win(self):
        return self.how_over == HowOver.WIN

    def is_draw(self):
        return self.how_over == HowOver.DRAW

    def is_winner(self, player):
        assert (player == chess.WHITE or player == chess.BLACK)

        if self.how_over == HowOver.WIN:
            return self.who_is_winner == player
        else:
            return False

    def print_info(self):
        print('over_event:', 'how_over:', self.how_over, 'who_is_winner:', self.who_is_winner)

    def test(self):
        if self.how_over == HowOver.WIN:
            assert (self.who_is_winner is not None)
            assert (self.who_is_winner is chess.WHITE or self.who_is_winner is chess.BLACK)
        if self.how_over == HowOver.DRAW:
            assert (self.who_is_winner is None)
