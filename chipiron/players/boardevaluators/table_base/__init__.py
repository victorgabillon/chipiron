"""
This module provides functionality for creating and using table bases for board evaluation.

Table bases are precomputed endgame databases that store optimal moves for every possible position in a specific
 endgame scenario. These databases can be used to improve the performance of board evaluation algorithms
  by providing accurate evaluations for endgame positions.

This module exports the following functions and classes:
- create_syzygy: A function for creating a SyzygyTable object.
- SyzygyTable: A class representing a table base for endgame evaluation.
"""

from .syzygy_table import SyzygyTable

__all__ = ["SyzygyTable"]
