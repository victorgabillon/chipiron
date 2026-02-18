"""Runtime-configurable addons for search dynamics wrappers."""

from dataclasses import dataclass
from enum import Enum


class SearchDynamicsAddonType(str, Enum):
    """Supported optional search dynamics wrappers."""

    CHESS_COPY_STACK = "CHESS_COPY_STACK"


@dataclass(frozen=True)
class ChessCopyStackAddonArgs:
    """Configuration for depth-aware chess board stack copying."""

    type: SearchDynamicsAddonType = SearchDynamicsAddonType.CHESS_COPY_STACK
    copy_stack_until_depth: int = 2
    deep_copy_legal_moves: bool = True


# Union-style alias for addon args (extend with `| OtherAddonArgs` later).
SearchDynamicsAddonArgs = ChessCopyStackAddonArgs

