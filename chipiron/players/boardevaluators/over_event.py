import chess


class OverEvent:
    HOW_OVERS = [WIN, DRAW, DO_NOT_KNOW_OVER] = [1, 2, 3]
    WINNERS = [WHITE, BLACK, NO_KNOWN_WINNER] = [chess.WHITE, chess.BLACK, None]

    def __init__(self, how_over=DO_NOT_KNOW_OVER, who_is_winner=NO_KNOWN_WINNER):
        assert how_over in self.HOW_OVERS
        assert who_is_winner in self.WINNERS
        self.how_over = how_over
        self.who_is_winner = who_is_winner
        if how_over == self.WIN:
            assert (self.who_is_winner is self.WHITE or self.who_is_winner is self.BLACK)
        elif how_over == self.DRAW:
            assert (self.who_is_winner is self.NO_KNOWN_WINNER)

    def becomes_over(self, how_over, who_is_winner=NO_KNOWN_WINNER):
        self.how_over = how_over
        self.who_is_winner = who_is_winner
        assert how_over in self.HOW_OVERS
        assert who_is_winner in self.WINNERS

    def simple_string(self):
        if self.how_over == self.WIN:
            if self.who_is_winner == chess.WHITE:
                return 'Win-Wh'
            elif self.who_is_winner == chess.BLACK:
                return 'Win-Bl'
            else:
                raise Exception('error: winner is not properly defined.')
        elif self.how_over == self.DRAW:
            return 'Draw'
        elif self.how_over == self.DO_NOT_KNOW_OVER:
            return '?'
        else:
            raise Exception('error: over is not properly defined.')

    def __bool__(self):
        assert (1 == 0)

    def is_over(self):
        return self.how_over == self.WIN or self.how_over == self.DRAW

    def is_win(self):
        return self.how_over == self.WIN

    def is_draw(self):
        return self.how_over == self.DRAW

    def is_winner(self, player):
        assert (player == chess.WHITE or player == chess.BLACK)

        if self.how_over == self.WIN:
            return self.who_is_winner == player
        else:
            return False

    def print_info(self):
        print('over_event:', 'how_over:', self.how_over, 'who_is_winner:', self.who_is_winner)

    def test(self):

        if self.how_over == self.WIN:
            assert (self.who_is_winner is not None)
            assert (self.who_is_winner is chess.WHITE or self.who_is_winner is chess.BLACK)
        if self.how_over == self.DRAW:
            assert (self.who_is_winner is None)
