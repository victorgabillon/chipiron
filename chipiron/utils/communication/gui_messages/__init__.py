"""
Module containing the different messages that can be sent to the GUI.
"""
from .evaluation_messsage import EvaluationMessage
from .game_status_message import GameStatusMessage, BackMessage

__all__ = [
    "GameStatusMessage",
    "EvaluationMessage",
    "BackMessage"
]
