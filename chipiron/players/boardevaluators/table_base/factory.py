"""
Module to create a SyzygyTable object if the path to the syzygy tables exists,
otherwise return None.
"""
import os

from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.utils import path


def create_syzygy() -> SyzygyTable | None:
    """
    Create a SyzygyTable object if the path to the syzygy tables exists,
    otherwise return None.

    Returns:
        SyzygyTable | None: The created SyzygyTable object or None if the path does not exist.
    """
    syzygy_table: SyzygyTable | None
    path_to_table: path = "data/syzygy-tables/"
    is_exist: bool = os.path.exists(path_to_table)

    if is_exist:
        syzygy_table = SyzygyTable(path_to_table=path_to_table)
    else:
        print('WARNING: no folder found for syzygy tables so NOT using it')
        syzygy_table = None

    return syzygy_table
