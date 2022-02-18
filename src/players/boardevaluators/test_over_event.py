import chess
from src.players.boardevaluators.over_event import OverEvent
import unittest


class TestOverEvent(unittest.TestCase):

    def test_construct_over_events(self):


        def test_construct_over_event( test_over_event):
            if test_over_event.how_over:
                self.assertTrue(test_over_event.is_win or test_over_event.is_draw)
            if test_over_event.is_win:
                self.assertTrue(not test_over_event.is_draw)
                self.assertTrue(test_over_event.how_over.who_is_winner is not None)
                self.assertTrue(
                    test_over_event.who_is_winner is chess.WHITE or test_over_event.who_is_winner is chess.BLACK)
            if test_over_event.is_draw:
                self.assertTrue(not test_over_event.is_win)
                self.assertTrue(test_over_event.who_is_winner is None)

        test_over_event = OverEvent()
        test_construct_over_event(test_over_event)

        test_over_event = OverEvent(True)
        test_construct_over_event(test_over_event)