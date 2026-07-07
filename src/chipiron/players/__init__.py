"""Module for players in the game."""

from __future__ import annotations

__all__ = [
    "GamePlayer",
    "InProcessPlayerHandle",
    "Player",
    "PlayerArgs",
    "PlayerFactoryArgs",
    "PlayerHandle",
    "PlayerProcess",
    "StockfishSelectorArgs",
]


def __getattr__(name: str) -> object:
    """Load public player symbols lazily to keep narrow imports lightweight."""
    if name == "GamePlayer":
        from .game_player import GamePlayer

        return GamePlayer
    if name == "StockfishSelectorArgs":
        from .move_selector.stockfish_args import StockfishSelectorArgs

        return StockfishSelectorArgs
    if name == "Player":
        from .player import Player

        return Player
    if name in {"PlayerArgs", "PlayerFactoryArgs"}:
        from .player_args import PlayerArgs, PlayerFactoryArgs

        return {"PlayerArgs": PlayerArgs, "PlayerFactoryArgs": PlayerFactoryArgs}[name]
    if name in {"InProcessPlayerHandle", "PlayerHandle"}:
        from .player_handle import InProcessPlayerHandle, PlayerHandle

        return {
            "InProcessPlayerHandle": InProcessPlayerHandle,
            "PlayerHandle": PlayerHandle,
        }[name]
    if name == "PlayerProcess":
        from .player_thread import PlayerProcess

        return PlayerProcess
    raise AttributeError(name)
