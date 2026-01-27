"""Tree-and-value move selector args with game-specific evaluator inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from anemone import TreeAndValuePlayerArgs as AnemoneTreeArgs
from valanga import TurnState

from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes

StateT = TypeVar("StateT", bound=TurnState)
EvalArgsT = TypeVar("EvalArgsT")

TREE_AND_VALUE_LITERAL_STRING: Literal["TreeAndValue"] = "TreeAndValue"


@dataclass(frozen=True)
class TreeAndValueAppArgs(Generic[StateT, EvalArgsT]):
    """Generic wrapper for tree-and-value settings plus evaluator args."""

    anemone_args: AnemoneTreeArgs
    evaluator_args: EvalArgsT

    @property
    def type(self) -> MoveSelectorTypes:
        return MoveSelectorTypes.TreeAndValue
