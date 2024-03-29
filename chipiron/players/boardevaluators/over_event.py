from dataclasses import dataclass
from enum import Enum

import chess


class HowOver(Enum):
    WIN = 1
    DRAW = 2
    DO_NOT_KNOW_OVER = 3


class Winner(Enum):
    WHITE = chess.WHITE
    BLACK = chess.BLACK
    NO_KNOWN_WINNER = None

    def is_none(self) -> bool:
        return self is None

    def is_white(self) -> bool:
        if not self.is_none():
            return bool(self) is chess.WHITE
        else:
            return False

    def is_black(self) -> bool:
        if not self.is_none():
            return bool(self) is chess.BLACK
        else:
            return False


class OverTags(str, Enum):
    TAG_WIN_WHITE = 'Win-Wh'
    TAG_WIN_BLACK = 'Win-Bl'
    TAG_DRAW = 'Draw'
    TAG_DO_NOT_KNOW = '?'


@dataclass(slots=True)
class OverEvent:
    how_over: HowOver = HowOver.DO_NOT_KNOW_OVER
    who_is_winner: Winner = Winner.NO_KNOWN_WINNER

    def __post_init__(self) -> None:
        assert self.how_over in HowOver
        assert self.who_is_winner in Winner

        if self.how_over == HowOver.WIN:
            assert (self.who_is_winner is Winner.WHITE or self.who_is_winner is Winner.BLACK)
        elif self.how_over == HowOver.DRAW:
            assert (self.who_is_winner is Winner.NO_KNOWN_WINNER)

    def becomes_over(
            self,
            how_over: HowOver,
            who_is_winner: Winner = Winner.NO_KNOWN_WINNER
    ) -> None:
        # FIXME its it just replacing an over event by another over event?? should we simplify in some way?
        self.how_over = how_over
        self.who_is_winner = who_is_winner

    def get_over_tag(self) -> OverTags:
        """ returns a simple string that is used a tag in the databases"""
        if self.how_over == HowOver.WIN:
            if self.who_is_winner.is_white():
                return OverTags.TAG_WIN_WHITE
            elif self.who_is_winner.is_black():
                return OverTags.TAG_WIN_BLACK
            else:
                raise Exception('error: winner is not properly defined.')
        elif self.how_over == HowOver.DRAW:
            return OverTags.TAG_DRAW
        elif self.how_over == HowOver.DO_NOT_KNOW_OVER:
            return OverTags.TAG_DO_NOT_KNOW
        else:
            raise Exception('error: over is not properly defined.')

    def __bool__(self) -> None:
        raise Exception('Nooooooooooo  in over ebvent.py')

    def is_over(self) -> bool:
        return self.how_over == HowOver.WIN or self.how_over == HowOver.DRAW

    def is_win(self) -> bool:
        return self.how_over == HowOver.WIN

    def is_draw(self) -> bool:
        return self.how_over == HowOver.DRAW

    def is_winner(
            self,
            player: chess.Color
    ) -> bool:
        assert (player == chess.WHITE or player == chess.BLACK)

        if self.how_over == HowOver.WIN:
            if (self.who_is_winner == Winner.WHITE and player == chess.WHITE
                    or self.who_is_winner == Winner.BLACK and player == chess.BLACK):
                return True
            else:
                return False
        else:
            return False

    def print_info(self) -> None:
        print('over_event:', 'how_over:', self.how_over, 'who_is_winner:', self.who_is_winner)

    def test(self) -> None:
        if self.how_over == HowOver.WIN:
            assert (self.who_is_winner is not None)
            assert (self.who_is_winner.is_white() or self.who_is_winner.is_black())
        if self.how_over == HowOver.DRAW:
            assert (self.who_is_winner is None)
