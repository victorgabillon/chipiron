"""init for chipiron."""

from anemone.node_selector.node_selector_types import NodeSelectorType

from . import games as game
from . import players as player
from . import utils as tool
from .utils.my_random import set_seeds


def _patch_anemone_runtime_type_hints() -> None:
	"""Bind missing runtime names required by parsley/get_type_hints."""
	from anemone.node_selector.linoo import linoo

	if not hasattr(linoo, "NodeSelectorType"):
		linoo.NodeSelectorType = NodeSelectorType


_patch_anemone_runtime_type_hints()

__all__ = ["game", "player", "set_seeds", "tool"]
