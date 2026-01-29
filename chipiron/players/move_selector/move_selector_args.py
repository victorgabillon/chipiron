"""
This module defines the MoveSelectorArgs protocol and helpers for move selector arguments.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, TypeAlias, cast

from anemone import TreeAndValuePlayerArgs
from dacite import Config, from_dict

from . import human, random, stockfish
from .move_selector_types import MoveSelectorTypes


class MoveSelectorArgs(Protocol):
    """Protocol for arguments for MoveSelector construction."""

    type: MoveSelectorTypes

    def is_human(self) -> bool:
        return self.type.is_human()


AnyMoveSelectorArgs: TypeAlias = (
    TreeAndValuePlayerArgs
    | human.CommandLineHumanPlayerArgs
    | human.GuiHumanPlayerArgs
    | random.Random
    | stockfish.StockfishPlayer
)

_MOVE_SELECTOR_ARGS_TYPES = (
    TreeAndValuePlayerArgs,
    human.CommandLineHumanPlayerArgs,
    human.GuiHumanPlayerArgs,
    random.Random,
    stockfish.StockfishPlayer,
)

_MOVE_SELECTOR_ARGS_BY_TYPE: dict[MoveSelectorTypes, type[Any]] = {
    MoveSelectorTypes.TreeAndValue: TreeAndValuePlayerArgs,
    MoveSelectorTypes.CommandLineHuman: human.CommandLineHumanPlayerArgs,
    MoveSelectorTypes.GuiHuman: human.GuiHumanPlayerArgs,
    MoveSelectorTypes.Random: random.Random,
    MoveSelectorTypes.Stockfish: stockfish.StockfishPlayer,
}


def resolve_move_selector_args(raw: Any) -> AnyMoveSelectorArgs:
    """Resolve move selector args from parsed YAML/dict inputs."""
    if isinstance(raw, _MOVE_SELECTOR_ARGS_TYPES):
        return raw
    if is_dataclass(raw) and hasattr(raw, "type"):
        return cast("AnyMoveSelectorArgs", raw)
    if isinstance(raw, Mapping):
        selector_type = raw.get("type")
        if selector_type is None:
            raise ValueError("Move selector args must include a 'type' field.")
        move_selector_type = (
            selector_type
            if isinstance(selector_type, MoveSelectorTypes)
            else MoveSelectorTypes(str(selector_type))
        )
        target_cls = _MOVE_SELECTOR_ARGS_BY_TYPE[move_selector_type]
        return cast(
            "AnyMoveSelectorArgs",
            from_dict(
                data_class=target_cls,
                data=dict(raw),
                config=Config(cast=[Enum]),
            ),
        )
    raise TypeError(f"Unsupported move selector args payload: {raw!r}")
